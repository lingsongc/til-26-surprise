"""Survival-first deterministic agent for TIL-26 Surprise.

Drop-in replacement for participant/src/algo_agent.py (keeps the AlgoAgent
class name that server.py imports — nothing else needs wiring).

Architecture ported from our main-challenge AEManager design:
  - WorldMemory  : persistent cross-turn state. The observation is stateless
                   (fog of war has no memory, production queues are invisible),
                   so we remember terrain, our own pending unit orders, and
                   which players we've already proposed peace to.
  - Turn pipeline: diplomacy -> construction -> production -> unit orders,
                   sharing one `planned` tile set so our own actions never
                   collide with each other.
  - Hardened     : any exception => empty (no-op) payload; never a crash.
                   Invalid actions are silent no-ops in this engine, so we
                   can be liberal — the only fatal mistakes are crashing or
                   blowing the 10s deadline.

Strategy: surviving to max_turns is a (co-)win. We never need to conquer
anyone; we need to still own one completed Base at the end. So:
  1. redundant Bases (spare lives) before anything else,
  2. economy (Barracks early, Mines on rich tiles, Factory later),
  3. peace with everyone we meet (free immunity until the turn-200 cutoff),
  4. a defensive ring of Infantry + Medics, Artillery in the late game.

We deliberately never read global_chat / private_chat — a deterministic agent
is immune to prompt injection, and ignoring chat costs us nothing.
"""

from __future__ import annotations

from agent_base import PlayerAgent
from engine.actions import (
    ActionPayload,
    AttackAction,
    ConstructBuildingAction,
    MoveAction,
    ProduceUnitAction,
    ProposeTreatyAction,
    RespondTreatyAction,
)
from engine.constants import BUILDING_STATS, TREATY_CUTOFF_TURN, UNIT_STATS
from engine.hex_grid import HexCoord, HexGrid

_BUILDING_TYPES = frozenset(BUILDING_STATS)  # names that are buildings
_PRODUCTION_TYPES = frozenset(("Barracks", "Factory", "Airbase"))

# ── tuning knobs ──────────────────────────────────────────────────────────────
BASE_TARGET_EARLY = 2      # Bases (incl. under construction) before the switch turn
BASE_TARGET_LATE = 3
BASE_TARGET_SWITCH_TURN = 80
MINE_CAP = 4               # max Mines we ever build
FACTORY_TURN = 45          # earliest turn we consider a Factory
ARTILLERY_CAP = 3
ARMY_CAP = 18              # max combat units (alive + queued)
MEDIC_RATIO = 5            # aim for 1 Medic per this many combat units
PENDING_PER_BUILDING = 3   # max queued units per production building
THREAT_RADIUS = 6          # enemy units this close to a Base are "threats"
CREEP_RADIUS = 4           # enemy buildings this close to a Base get attacked
PROPOSE_EVERY = 15         # re-propose peace to a player every N turns


# ── persistent memory ─────────────────────────────────────────────────────────


class WorldMemory:
    """Everything worth remembering across turns (the obs alone forgets it all)."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.last_turn: int = -1
        self.terrain: dict[tuple[int, int], str] = {}
        # building_id -> list of (unit_type, ready_turn) we have queued there
        self.pending: dict[str, list[tuple[str, int]]] = {}
        self.last_proposed: dict[str, int] = {}  # player_id -> turn we proposed

    def update(self, obs: dict) -> None:
        turn = obs.get("turn_number", 0)
        if turn < self.last_turn:  # same server process, new game
            self.reset()
        self.last_turn = turn
        for tile in obs.get("visible_tiles", []):
            self.terrain[(tile["q"], tile["r"])] = tile.get("terrain", "normal")
        # prune production orders that should have spawned by now
        for bid in list(self.pending):
            self.pending[bid] = [(u, t) for (u, t) in self.pending[bid] if t > turn]
            if not self.pending[bid]:
                del self.pending[bid]

    def pending_count(self, building_id: str) -> int:
        return len(self.pending.get(building_id, []))

    def pending_units(self) -> list[str]:
        return [u for orders in self.pending.values() for (u, _) in orders]

    def note_production(self, building_id: str, unit_type: str, turn: int) -> None:
        ready = turn + UNIT_STATS[unit_type].build_turns
        self.pending.setdefault(building_id, []).append((unit_type, ready))


# ── the agent ─────────────────────────────────────────────────────────────────


class AlgoAgent(PlayerAgent):
    def __init__(self) -> None:
        self.mem = WorldMemory()

    async def decide(self, observation: dict) -> ActionPayload:
        pid = observation.get("player_id", "unknown")
        turn = observation.get("turn_number", 0)
        try:
            actions = self._decide(observation, pid, turn)
        except Exception:
            actions = []  # a bad turn must never become a crashed turn
        return ActionPayload(player_id=pid, turn_number=turn, actions=actions)

    # ── main pipeline ─────────────────────────────────────────────────────────

    def _decide(self, obs: dict, pid: str, turn: int) -> list:
        self.mem.update(obs)
        grid = HexGrid(obs.get("map_width", 35), obs.get("map_height", 30))
        gold = obs.get("resources", {}).get("gold", 0)

        # players we are at peace with (ACTIVE or BREAKING both block attacks)
        allies: set[str] = {
            t.get("partner_id") for t in obs.get("treaties", []) if t.get("partner_id")
        }

        scan = self._scan(obs, pid, allies)
        own_units, own_buildings, enemy_units, enemy_buildings, friendly_coords, occupied = scan

        # tiles our own actions claim this turn (builds, moves, spawn targets)
        planned: set[tuple[int, int]] = set(occupied)

        actions: list = []
        actions += self._diplomacy(obs, turn, allies)
        actions += self._construction(
            obs, turn, gold, grid, own_buildings,
            enemy_units, enemy_buildings, planned,
        )
        gold_left = gold - sum(
            BUILDING_STATS[a.building_type].gold_cost
            for a in actions
            if isinstance(a, ConstructBuildingAction)
        )
        actions += self._production(
            turn, gold_left, grid, own_units, own_buildings, planned,
            reserving=self._base_reserve(turn, own_buildings, actions),
        )
        actions += self._unit_orders(
            grid, own_units, own_buildings,
            enemy_units, enemy_buildings, friendly_coords, planned,
        )
        return actions

    # ── observation parsing ───────────────────────────────────────────────────

    @staticmethod
    def _scan(obs: dict, pid: str, allies: set[str]):
        own_units: list[dict] = []
        own_buildings: list[dict] = []
        enemy_units: list[dict] = []
        enemy_buildings: list[dict] = []
        friendly_coords: set[tuple[int, int]] = set()  # ours + allies (splash safety)
        occupied: set[tuple[int, int]] = set()
        for tile in obs.get("visible_tiles", []):
            for e in tile.get("entities", []):
                key = (e["q"], e["r"])
                occupied.add(key)
                owner = e.get("owner_id")
                is_building = e.get("type") in _BUILDING_TYPES
                if owner == pid:
                    friendly_coords.add(key)
                    (own_buildings if is_building else own_units).append(e)
                elif owner in allies:
                    friendly_coords.add(key)  # never attack, never splash
                else:
                    (enemy_buildings if is_building else enemy_units).append(e)
        return own_units, own_buildings, enemy_units, enemy_buildings, friendly_coords, occupied

    # ── stage 0: diplomacy ────────────────────────────────────────────────────

    def _diplomacy(self, obs: dict, turn: int, allies: set[str]) -> list:
        if turn >= TREATY_CUTOFF_TURN:
            return []  # diplomacy is closed; anything we send is ignored
        actions: list = []
        for prop in obs.get("incoming_treaty_proposals", []):
            proposer = prop.get("proposer_id", "")
            if proposer:
                actions.append(
                    RespondTreatyAction(proposing_player_id=proposer, accept=True)
                )
                allies.add(proposer)  # active this turn; stop targeting them now
        for other in obs.get("known_players", []):
            if other in allies:
                continue
            if turn - self.mem.last_proposed.get(other, -(10 ** 6)) >= PROPOSE_EVERY:
                actions.append(ProposeTreatyAction(target_player_id=other))
                self.mem.last_proposed[other] = turn
        return actions

    # ── stage 1+2: construction (bases, then economy) ─────────────────────────

    @staticmethod
    def _base_target(turn: int) -> int:
        return BASE_TARGET_LATE if turn >= BASE_TARGET_SWITCH_TURN else BASE_TARGET_EARLY

    def _base_reserve(self, turn: int, own_buildings: list[dict], actions: list) -> int:
        """Gold we refuse to spend on anything but the next Base."""
        bases = sum(1 for b in own_buildings if b["type"] == "Base")
        bases += sum(
            1 for a in actions
            if isinstance(a, ConstructBuildingAction) and a.building_type == "Base"
        )
        return BUILDING_STATS["Base"].gold_cost if bases < self._base_target(turn) else 0

    def _construction(
        self, obs, turn, gold, grid,
        own_buildings, enemy_units, enemy_buildings, planned,
    ) -> list:
        actions: list = []
        complete = [b for b in own_buildings if b.get("is_complete")]
        bases = [b for b in own_buildings if b["type"] == "Base"]
        enemy_coords = [
            HexCoord(e["q"], e["r"]) for e in enemy_units + enemy_buildings
        ]

        # 1) Base redundancy — the single biggest survival lever
        if len(bases) < self._base_target(turn) and gold >= BUILDING_STATS["Base"].gold_cost:
            spot = self._pick_base_tile(obs, grid, bases, enemy_coords, planned)
            if spot:
                actions.append(
                    ConstructBuildingAction(building_type="Base", coord=HexCoord(*spot))
                )
                planned.add(spot)
                gold -= BUILDING_STATS["Base"].gold_cost
                bases.append({"q": spot[0], "r": spot[1], "type": "Base"})

        # while still below the Base target, protect 300g for the next one
        reserve = BUILDING_STATS["Base"].gold_cost if len(bases) < self._base_target(turn) else 0

        def afford(cost: int) -> bool:
            return gold - cost >= reserve

        # 2) one Barracks as early as possible
        if not any(b["type"] == "Barracks" for b in own_buildings) and afford(100):
            spot = self._pick_adjacent_tile(grid, complete, own_buildings, planned, prefer_rich=False)
            if spot:
                actions.append(ConstructBuildingAction(building_type="Barracks", coord=HexCoord(*spot)))
                planned.add(spot)
                gold -= BUILDING_STATS["Barracks"].gold_cost

        # 3) Mines — rich tiles (50g/turn) strongly preferred over normal (20g/turn)
        mines = sum(1 for b in own_buildings if b["type"] == "Mine")
        if mines < MINE_CAP and afford(200):
            spot = self._pick_adjacent_tile(grid, complete, own_buildings, planned, prefer_rich=True)
            if spot:
                actions.append(ConstructBuildingAction(building_type="Mine", coord=HexCoord(*spot)))
                planned.add(spot)
                gold -= BUILDING_STATS["Mine"].gold_cost

        # 4) one Factory for Artillery once the core economy stands
        if (
            turn >= FACTORY_TURN
            and not any(b["type"] == "Factory" for b in own_buildings)
            and afford(300)
        ):
            spot = self._pick_adjacent_tile(grid, complete, own_buildings, planned, prefer_rich=False)
            if spot:
                actions.append(ConstructBuildingAction(building_type="Factory", coord=HexCoord(*spot)))
                planned.add(spot)

        return actions

    def _pick_base_tile(self, obs, grid, bases, enemy_coords, planned):
        """Best visible empty tile for a new Base: rich > spread out > away from enemies."""
        base_coords = [HexCoord(b["q"], b["r"]) for b in bases]
        best, best_score = None, -(10 ** 9)
        for tile in obs.get("visible_tiles", []):
            key = (tile["q"], tile["r"])
            if key in planned:
                continue
            c = HexCoord(*key)
            d_enemy = min((grid.distance(c, e) for e in enemy_coords), default=99)
            if d_enemy < 3:
                continue  # don't found a Base under an enemy's nose
            d_own = min((grid.distance(c, b) for b in base_coords), default=0)
            if base_coords and d_own < 2:
                continue  # spread: one artillery shot must not splash two Bases
            score = 0.0
            if tile.get("terrain") == "rich_resource":
                score += 40.0  # 50g/turn instead of 10g/turn
            score += 4.0 * min(d_own, 6)   # spread out, with diminishing returns
            score += min(d_enemy, 8)       # prefer quieter ground
            if score > best_score:
                best, best_score = key, score
        return best

    def _pick_adjacent_tile(self, grid, complete, own_buildings, planned, prefer_rich):
        """Free tile adjacent to a completed building; never chokes a production
        building's spawn ring (it must keep >= 2 free neighbours)."""
        prod_coords = [
            HexCoord(b["q"], b["r"]) for b in own_buildings
            if b["type"] in _PRODUCTION_TYPES
        ]

        def free_neighbours(c: HexCoord, minus: tuple[int, int]) -> int:
            return sum(
                1 for nb in grid.neighbors(c)
                if (nb.q, nb.r) not in planned and (nb.q, nb.r) != minus
            )

        best, best_score = None, -(10 ** 9)
        for b in complete:
            anchor = HexCoord(b["q"], b["r"])
            for nb in grid.neighbors(anchor):
                key = (nb.q, nb.r)
                if key in planned:
                    continue
                if any(
                    grid.distance(nb, pc) <= 1 and free_neighbours(pc, key) < 2
                    for pc in prod_coords
                ):
                    continue  # would box in a Barracks/Factory (units would be lost)
                terrain = self.mem.terrain.get(key, "normal")
                score = 50.0 if (prefer_rich and terrain == "rich_resource") else 0.0
                if prefer_rich and terrain != "rich_resource":
                    score -= 1.0  # still allowed, just deprioritised
                if score > best_score:
                    best, best_score = key, score
        return best

    # ── stage 3: production ───────────────────────────────────────────────────

    def _production(self, turn, gold, grid, own_units, own_buildings, planned, reserving) -> list:
        actions: list = []
        pending = self.mem.pending_units()
        combat = sum(1 for u in own_units if u["type"] != "Medic")
        combat += sum(1 for u in pending if u != "Medic")
        medics = sum(1 for u in own_units if u["type"] == "Medic")
        medics += sum(1 for u in pending if u == "Medic")
        army_target = min(ARMY_CAP, 3 + turn // 10)

        def afford(cost: int) -> bool:
            return gold - cost >= reserving

        def spawn_target(b: dict) -> HexCoord:
            anchor = HexCoord(b["q"], b["r"])
            for nb in grid.neighbors(anchor):
                if (nb.q, nb.r) not in planned:
                    return nb
            return grid.neighbors(anchor)[0]  # engine falls back to any free ring tile

        # Artillery from the Factory — the best base-defence unit in the game
        artillery = sum(1 for u in own_units if u["type"] == "Artillery")
        artillery += sum(1 for u in pending if u == "Artillery")
        for f in (b for b in own_buildings if b["type"] == "Factory" and b.get("is_complete")):
            if artillery >= ARTILLERY_CAP or not afford(200):
                break
            if self.mem.pending_count(f["id"]) >= PENDING_PER_BUILDING:
                continue
            actions.append(
                ProduceUnitAction(building_id=f["id"], unit_type="Artillery", target=spawn_target(f))
            )
            self.mem.note_production(f["id"], "Artillery", turn)
            gold -= 200
            artillery += 1

        # Infantry (+ the occasional Medic) from every completed Barracks
        for b in (x for x in own_buildings if x["type"] == "Barracks" and x.get("is_complete")):
            while (
                combat + medics < army_target
                and self.mem.pending_count(b["id"]) < PENDING_PER_BUILDING
            ):
                want_medic = medics * MEDIC_RATIO < combat
                unit = "Medic" if (want_medic and afford(100)) else "Infantry"
                cost = UNIT_STATS[unit].gold_cost
                if not afford(cost):
                    break
                actions.append(
                    ProduceUnitAction(building_id=b["id"], unit_type=unit, target=spawn_target(b))
                )
                self.mem.note_production(b["id"], unit, turn)
                gold -= cost
                if unit == "Medic":
                    medics += 1
                else:
                    combat += 1
        return actions

    # ── stage 4: unit orders ──────────────────────────────────────────────────

    def _unit_orders(
        self, grid, own_units, own_buildings,
        enemy_units, enemy_buildings, friendly_coords, planned,
    ) -> list:
        actions: list = []
        guard_points = [
            HexCoord(b["q"], b["r"])
            for b in own_buildings if b["type"] == "Base"
        ] or [HexCoord(b["q"], b["r"]) for b in own_buildings]
        if not guard_points:
            return actions

        def near_base(e: dict, radius: int) -> bool:
            c = HexCoord(e["q"], e["r"])
            return any(grid.distance(c, g) <= radius for g in guard_points)

        threats = [e for e in enemy_units if near_base(e, THREAT_RADIUS)]
        creep = [e for e in enemy_buildings if near_base(e, CREEP_RADIUS)]

        for u in own_units:
            here = HexCoord(u["q"], u["r"])

            if u["type"] == "Medic":
                mv = self._medic_move(grid, u, here, own_units, guard_points, planned)
                if mv:
                    actions.append(mv)
                continue

            # fire on anything in range (attacking never requires moving)
            target = self._pick_attack(
                grid, u, here, enemy_units, enemy_buildings, friendly_coords
            )
            if target is not None:
                actions.append(AttackAction(unit_id=u["id"], target=target))
                continue  # attackers hold ground

            # no shot: converge on the nearest threat, else hold near our Base
            goal = None
            pool = threats or creep
            if pool:
                nearest = min(pool, key=lambda e: grid.distance(here, HexCoord(e["q"], e["r"])))
                goal = HexCoord(nearest["q"], nearest["r"])
            else:
                home = min(guard_points, key=lambda g: grid.distance(here, g))
                if grid.distance(here, home) > 3:
                    goal = home  # drift back into the defensive ring
            if goal is not None:
                mv = self._move_toward(grid, u, here, goal, planned)
                if mv:
                    actions.append(mv)
        return actions

    def _pick_attack(self, grid, u, here, enemy_units, enemy_buildings, friendly_coords):
        ar = u.get("attack_range", 0)
        if ar < 1:
            return None
        is_artillery = u["type"] == "Artillery"

        def ok(e: dict) -> bool:
            c = HexCoord(e["q"], e["r"])
            d = grid.distance(here, c)
            if not (1 <= d <= ar):
                return False
            if is_artillery:
                # splash hits EVERYONE in the ring, including us and allies
                if d < 2:
                    return False  # at range 1 our own tile is in the splash ring
                if any((s.q, s.r) in friendly_coords for s in grid.ring(c, 1)):
                    return False
            return True

        units = sorted((e for e in enemy_units if ok(e)), key=lambda e: e.get("hp", 999))
        if units:  # kill the squishiest attacker first (no overkill prevention)
            return HexCoord(units[0]["q"], units[0]["r"])
        bldgs = sorted((e for e in enemy_buildings if ok(e)), key=lambda e: e.get("hp", 999))
        if bldgs:
            return HexCoord(bldgs[0]["q"], bldgs[0]["r"])
        return None

    def _medic_move(self, grid, u, here, own_units, guard_points, planned):
        wounded = [
            w for w in own_units
            if w["id"] != u["id"]
            and w["type"] not in ("Medic", "Fighter", "Bomber")  # heals ground only
            and w.get("hp", 0) < w.get("max_hp", 0)
            and grid.distance(here, HexCoord(w["q"], w["r"])) <= 5
        ]
        if wounded:
            tgt = min(wounded, key=lambda w: w.get("hp", 999))
            goal = HexCoord(tgt["q"], tgt["r"])
            if grid.distance(here, goal) <= 1:
                return None  # already adjacent — passive heal does the rest
            return self._move_toward(grid, u, here, goal, planned)
        home = min(guard_points, key=lambda g: grid.distance(here, g))
        if grid.distance(here, home) > 2:
            return self._move_toward(grid, u, here, home, planned)
        return None

    def _move_toward(self, grid, u, here, goal, planned):
        """Greedy multi-step path within the movement-point budget. Difficult
        terrain costs 2 to enter; unknown (fogged) tiles are assumed cost 1 —
        if we guess wrong the engine silently drops the move, which is fine."""
        budget = u.get("movement_range", 0)
        if budget < 1:
            return None
        path = [here]
        cur, remaining = here, budget
        while remaining > 0 and cur != goal:
            best, best_d, best_cost = None, grid.distance(cur, goal), 0
            for nb in grid.neighbors(cur):
                key = (nb.q, nb.r)
                if key in planned:
                    continue
                cost = 2 if self.mem.terrain.get(key) == "difficult" else 1
                if cost > remaining:
                    continue
                d = grid.distance(nb, goal)
                if d < best_d:
                    best, best_d, best_cost = nb, d, cost
            if best is None:
                break
            path.append(best)
            cur = best
            remaining -= best_cost
        if len(path) < 2:
            return None
        planned.add((path[-1].q, path[-1].r))  # claim the destination
        return MoveAction(unit_id=u["id"], path=path)
