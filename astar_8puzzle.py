import pygame
import heapq
import random
import time

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
WINDOW_W, WINDOW_H = 900, 620
TILE_SIZE         = 120
GRID_COLS         = 3
BOARD_OFFSET_X    = 60
BOARD_OFFSET_Y    = 160

GOAL_STATE = (1, 2, 3, 4, 5, 6, 7, 8, 0)   # 0 = blank

# Colours
BG          = (15,  23,  42)
TILE_CLR    = (30,  64, 175)
TILE_HOVER  = (37,  99, 235)
BLANK_CLR   = (30,  41,  59)
TEXT_CLR    = (255, 255, 255)
ACCENT      = (250, 204,  21)
GREEN       = ( 34, 197,  94)
RED         = (239,  68,  68)
GRAY        = (100, 116, 139)
PANEL_BG    = ( 30,  41,  59)
PATH_CLR    = ( 99, 102, 241)
EXPLORED_C  = ( 51,  65,  85)

MOVES = {
    'UP':    -3,
    'DOWN':   3,
    'LEFT':  -1,
    'RIGHT':  1,
}

# ─────────────────────────────────────────────
#  PUZZLE LOGIC
# ─────────────────────────────────────────────

def is_solvable(state):
    """Count inversions; puzzle solvable iff inversions is even."""
    tiles = [t for t in state if t != 0]
    inv = sum(1 for i in range(len(tiles))
                for j in range(i + 1, len(tiles))
                if tiles[i] > tiles[j])
    return inv % 2 == 0


def get_neighbors(state):
    neighbors = []
    idx = state.index(0)
    row, col = divmod(idx, 3)
    for move, delta in MOVES.items():
        new_idx = idx + delta
        if move == 'LEFT'  and col == 0: continue
        if move == 'RIGHT' and col == 2: continue
        if move == 'UP'    and row == 0: continue
        if move == 'DOWN'  and row == 2: continue
        s = list(state)
        s[idx], s[new_idx] = s[new_idx], s[idx]
        neighbors.append((tuple(s), move))
    return neighbors


def manhattan(state):
    dist = 0
    for idx, tile in enumerate(state):
        if tile == 0:
            continue
        goal_idx = GOAL_STATE.index(tile)
        dist += abs(idx // 3 - goal_idx // 3) + abs(idx % 3 - goal_idx % 3)
    return dist


def misplaced(state):
    return sum(1 for i, t in enumerate(state) if t != 0 and t != GOAL_STATE[i])


def astar(start, heuristic_fn):
    """Returns (solution_path, explored_states, stats)."""
    open_heap = []
    heapq.heappush(open_heap, (heuristic_fn(start), 0, start, [start]))
    visited   = {}   # state -> g_cost
    explored  = []

    while open_heap:
        f, g, state, path = heapq.heappop(open_heap)

        if state in visited and visited[state] <= g:
            continue
        visited[state] = g
        explored.append(state)

        if state == GOAL_STATE:
            return path, explored, {'moves': g, 'explored': len(explored)}

        for neighbor, _ in get_neighbors(state):
            new_g = g + 1
            if neighbor not in visited or visited[neighbor] > new_g:
                h = heuristic_fn(neighbor)
                heapq.heappush(open_heap, (new_g + h, new_g, neighbor, path + [neighbor]))

    return None, explored, {}


def random_puzzle():
    while True:
        tiles = list(range(9))
        random.shuffle(tiles)
        state = tuple(tiles)
        if is_solvable(state) and state != GOAL_STATE:
            return state


def best_explored_states(explored, solution, limit=6):
    """Return explored states that are closest to goal and not in the final path."""
    path_set = set(solution)
    candidates = [s for s in explored if s not in path_set]
    ranked = sorted(candidates, key=lambda s: (manhattan(s), misplaced(s)))
    return ranked[:limit]


# ─────────────────────────────────────────────
#  DRAWING HELPERS
# ─────────────────────────────────────────────

def draw_board(surface, state, offset_x, offset_y, highlight=None, scale=1.0):
    ts = int(TILE_SIZE * scale)
    font_big = pygame.font.SysFont("Arial", int(ts * 0.45), bold=True)

    for idx, tile in enumerate(state):
        r, c = divmod(idx, 3)
        x = offset_x + c * (ts + 6)
        y = offset_y + r * (ts + 6)
        rect = pygame.Rect(x, y, ts, ts)

        if tile == 0:
            pygame.draw.rect(surface, BLANK_CLR, rect, border_radius=12)
        else:
            colour = ACCENT if (highlight and idx in highlight) else TILE_CLR
            pygame.draw.rect(surface, colour, rect, border_radius=12)
            txt = font_big.render(str(tile), True, TEXT_CLR)
            surface.blit(txt, txt.get_rect(center=rect.center))


def draw_button(surface, rect, text, colour, text_colour=TEXT_CLR, font_size=20):
    pygame.draw.rect(surface, colour, rect, border_radius=10)
    font = pygame.font.SysFont("Arial", font_size, bold=True)
    txt  = font.render(text, True, text_colour)
    surface.blit(txt, txt.get_rect(center=rect.center))


def draw_text(surface, text, x, y, font_size=20, colour=TEXT_CLR, bold=False, center=False):
    font = pygame.font.SysFont("Arial", font_size, bold=bold)
    txt  = font.render(text, True, colour)
    r    = txt.get_rect(center=(x, y)) if center else pygame.Rect(x, y, 0, 0)
    surface.blit(txt, r if center else (x, y))


# ─────────────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────────────

def main():
    pygame.init()
    screen  = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption("A* 8-Puzzle Solver")
    clock   = pygame.time.Clock()

    # State
    puzzle        = random_puzzle()
    solution      = []
    explored      = []
    stats         = {}
    step_idx      = 0
    playing       = False
    solved        = False
    heuristic_lbl = "Manhattan"
    heuristic_fn  = manhattan
    speed         = 4          # steps per second
    last_step     = 0
    show_explored = False
    timeline_scroll = 0
    dragging_scrollbar = False
    drag_offset = 0

    # Layout
    board_x, board_y = 60, 140
    timeline_rect = pygame.Rect(460, 140, 160, 440)
    right_panel = pygame.Rect(640, 110, 240, 490)
    timeline_item_h = 136
    timeline_gap = 12
    timeline_stride = timeline_item_h + timeline_gap
    inner_timeline_h = timeline_rect.height - 46
    timeline_content_top = timeline_rect.y + 36

    # Button rects
    btn_random  = pygame.Rect(680, 160, 190, 44)
    btn_solve   = pygame.Rect(680, 216, 190, 44)
    btn_play    = pygame.Rect(680, 272, 90,  44)
    btn_step    = pygame.Rect(782, 272, 88,  44)
    btn_reset   = pygame.Rect(680, 328, 190, 44)
    btn_heur    = pygame.Rect(680, 400, 190, 44)
    btn_exp     = pygame.Rect(680, 456, 190, 44)

    def run_solver():
        nonlocal solution, explored, stats, step_idx, playing, solved, timeline_scroll
        path, expl, st = astar(puzzle, heuristic_fn)
        solution  = path if path else []
        explored  = expl
        stats     = st
        step_idx  = 0
        playing   = False
        solved    = bool(path)
        timeline_scroll = 0

    def clamp(value, low, high):
        return max(low, min(value, high))

    def ensure_step_visible():
        nonlocal timeline_scroll
        if not solution:
            timeline_scroll = 0
            return

        content_h = len(solution) * timeline_stride
        max_scroll = max(0, content_h - inner_timeline_h)
        step_top = step_idx * timeline_stride
        step_bottom = step_top + timeline_item_h

        if step_top < timeline_scroll:
            timeline_scroll = step_top
        elif step_bottom > timeline_scroll + inner_timeline_h:
            timeline_scroll = step_bottom - inner_timeline_h

        timeline_scroll = clamp(timeline_scroll, 0, max_scroll)

    while True:
        now = time.time()
        screen.fill(BG)

        # ── Title ──
        draw_text(screen, "A*  8-Puzzle  Solver", WINDOW_W // 2, 32,
                  font_size=34, bold=True, colour=ACCENT, center=True)
        draw_text(screen, "FAST-NUCES Karachi  ·  K23-0874  ·  Saad Zaidi",
                  WINDOW_W // 2, 68, font_size=16, colour=GRAY, center=True)

        # ── Divider ──
        pygame.draw.line(screen, GRAY, (40, 90), (WINDOW_W - 40, 90), 1)

        # ── Labels ──
        draw_text(screen, "Current Board", board_x + 186, 118,
                  font_size=18, bold=True, center=True)

        # ── Current board ──
        current_state = solution[step_idx] if solution else puzzle
        draw_board(screen, current_state, board_x, board_y)

        # Progress bar next to main board
        progress_track = pygame.Rect(40, board_y, 10, 372)
        pygame.draw.rect(screen, EXPLORED_C, progress_track, border_radius=4)
        if solution:
            pct = step_idx / max(len(solution) - 1, 1)
            fill_h = int(progress_track.height * pct)
            fill_rect = pygame.Rect(progress_track.x, progress_track.y, progress_track.width, fill_h)
            pygame.draw.rect(screen, GREEN, fill_rect, border_radius=4)

        # ── Timeline panel with scrollbar ──
        pygame.draw.rect(screen, PANEL_BG, timeline_rect, border_radius=14)
        draw_text(screen, "Solution Timeline", timeline_rect.x + 12, timeline_rect.y + 10,
                  font_size=15, bold=True, colour=ACCENT)

        content_h = len(solution) * timeline_stride
        max_scroll = max(0, content_h - inner_timeline_h)
        timeline_scroll = clamp(timeline_scroll, 0, max_scroll)

        clip_rect = pygame.Rect(timeline_rect.x + 10, timeline_content_top, timeline_rect.width - 28, inner_timeline_h)
        old_clip = screen.get_clip()
        screen.set_clip(clip_rect)

        for idx, s in enumerate(solution):
            item_y = timeline_content_top + idx * timeline_stride - int(timeline_scroll)
            if item_y + timeline_item_h < clip_rect.top or item_y > clip_rect.bottom:
                continue

            item_rect = pygame.Rect(timeline_rect.x + 10, item_y, timeline_rect.width - 28, timeline_item_h)
            is_active = idx == step_idx
            pygame.draw.rect(screen, (71, 85, 105) if is_active else (51, 65, 85), item_rect, border_radius=10)
            draw_text(screen, f"Step {idx}", item_rect.x + 8, item_rect.y + 8,
                      font_size=12, colour=GREEN if is_active else GRAY, bold=is_active)
            draw_board(screen, s, item_rect.x + 8, item_rect.y + 24, scale=0.30)

        screen.set_clip(old_clip)

        # Scrollbar
        scroll_track = pygame.Rect(timeline_rect.right - 14, timeline_content_top, 8, inner_timeline_h)
        pygame.draw.rect(screen, (51, 65, 85), scroll_track, border_radius=5)
        if max_scroll > 0:
            thumb_h = max(32, int((inner_timeline_h * inner_timeline_h) / content_h))
            thumb_range = inner_timeline_h - thumb_h
            thumb_y = scroll_track.y + int((timeline_scroll / max_scroll) * thumb_range)
        else:
            thumb_h = inner_timeline_h
            thumb_y = scroll_track.y
        scrollbar_thumb = pygame.Rect(scroll_track.x, thumb_y, scroll_track.width, thumb_h)
        pygame.draw.rect(screen, ACCENT if dragging_scrollbar else (148, 163, 184), scrollbar_thumb, border_radius=5)

        # ── Right panel ──
        pygame.draw.rect(screen, PANEL_BG, right_panel, border_radius=14)

        # Stat labels
        draw_text(screen, "Heuristic",      656, 126, font_size=13, colour=GRAY)
        draw_text(screen, heuristic_lbl,     656, 144, font_size=16, bold=True, colour=ACCENT)
        draw_text(screen, f"States explored:  {stats.get('explored', '-')}",
              656, 518, font_size=14, colour=GRAY)
        draw_text(screen, f"Solution moves:   {stats.get('moves', '-')}",
              656, 538, font_size=14, colour=GRAY)
        draw_text(screen, f"Step:  {step_idx + 1 if solution else 0} / {len(solution)}",
              656, 558, font_size=14, colour=GRAY)

        # ── Buttons ──
        draw_button(screen, btn_random, "New Puzzle",   (51, 65, 85))
        draw_button(screen, btn_solve,  "Solve (A*)",    (30, 64, 175))
        draw_button(screen, btn_play,   "Play"  if not playing else "Pause",
                    GREEN if not playing else (202, 138, 4), text_colour=(0, 0, 0))
        draw_button(screen, btn_step,   "Next",           (51, 65, 85))
        draw_button(screen, btn_reset,  "Reset",         (127, 29, 29))
        draw_button(screen, btn_heur,
                    f"Heuristic: {'Manhattan' if heuristic_lbl == 'Manhattan' else 'Misplaced'}",
                    (55, 48, 163), font_size=15)
        draw_button(screen, btn_exp,
                    f"Explored: {'ON' if show_explored else 'OFF'}",
                    (20, 83, 45) if show_explored else (51, 65, 85), font_size=15)

        draw_text(screen, "Next = advance 1 move", 656, 380, font_size=12, colour=GRAY)

        # ── Explored preview in side panel ──
        if show_explored and explored:
            preview_rect = pygame.Rect(656, 402, 208, 106)
            pygame.draw.rect(screen, (51, 65, 85), preview_rect, border_radius=10)
            draw_text(screen, "Best explored (near goal)", preview_rect.x + 8, preview_rect.y + 6,
                      font_size=12, colour=GRAY)
            cards = best_explored_states(explored, solution, limit=3)
            for i, state in enumerate(cards):
                cx = preview_rect.x + 8 + i * 66
                cy = preview_rect.y + 24
                card = pygame.Rect(cx, cy, 60, 72)
                pygame.draw.rect(screen, (71, 85, 105), card, border_radius=8)
                draw_board(screen, state, cx + 4, cy + 4, scale=0.14)
                draw_text(screen, f"h={manhattan(state)}", cx + 6, cy + 58, font_size=11, colour=ACCENT)

        # ── Auto-play ──
        if playing and solution and step_idx < len(solution) - 1:
            if now - last_step > 1.0 / speed:
                step_idx += 1
                last_step = now
                if step_idx == len(solution) - 1:
                    playing = False

        ensure_step_visible()

        # ── Goal badge ──
        if solution and step_idx == len(solution) - 1:
            badge = pygame.Rect(BOARD_OFFSET_X, BOARD_OFFSET_Y - 36, 380, 30)
            pygame.draw.rect(screen, GREEN, badge, border_radius=8)
            draw_text(screen, "✓  Goal Reached!", BOARD_OFFSET_X + 190,
                      BOARD_OFFSET_Y - 21, font_size=16, bold=True,
                      colour=(0, 0, 0), center=True)

        pygame.display.flip()
        clock.tick(60)

        # ── Events ──
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return

            if event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos

                if scrollbar_thumb.collidepoint(mx, my):
                    dragging_scrollbar = True
                    drag_offset = my - scrollbar_thumb.y
                    continue

                if clip_rect.collidepoint(mx, my) and solution:
                    rel_y = my - timeline_content_top + int(timeline_scroll)
                    clicked_idx = rel_y // timeline_stride
                    if 0 <= clicked_idx < len(solution):
                        step_idx = clicked_idx
                        playing = False
                        ensure_step_visible()
                        continue

                if btn_random.collidepoint(mx, my):
                    puzzle   = random_puzzle()
                    solution = []; explored = []; stats = {}
                    step_idx = 0; playing = False; solved = False; timeline_scroll = 0

                elif btn_solve.collidepoint(mx, my):
                    run_solver()

                elif btn_play.collidepoint(mx, my):
                    if solution:
                        playing   = not playing
                        last_step = time.time()

                elif btn_step.collidepoint(mx, my):
                    if solution and step_idx < len(solution) - 1:
                        step_idx += 1

                elif btn_reset.collidepoint(mx, my):
                    step_idx = 0; playing = False; timeline_scroll = 0

                elif btn_heur.collidepoint(mx, my):
                    if heuristic_lbl == "Manhattan":
                        heuristic_lbl = "Misplaced"
                        heuristic_fn  = misplaced
                    else:
                        heuristic_lbl = "Manhattan"
                        heuristic_fn  = manhattan
                    solution = []; explored = []; stats = {}
                    step_idx = 0; playing = False; timeline_scroll = 0

                elif btn_exp.collidepoint(mx, my):
                    show_explored = not show_explored

            if event.type == pygame.MOUSEBUTTONUP:
                dragging_scrollbar = False

            if event.type == pygame.MOUSEMOTION and dragging_scrollbar and max_scroll > 0:
                target_y = event.pos[1] - drag_offset
                min_y = scroll_track.y
                max_y = scroll_track.bottom - scrollbar_thumb.height
                target_y = clamp(target_y, min_y, max_y)
                ratio = (target_y - min_y) / max(1, (max_y - min_y))
                timeline_scroll = ratio * max_scroll

            if event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if timeline_rect.collidepoint(mx, my) and max_scroll > 0:
                    timeline_scroll -= event.y * 42
                    timeline_scroll = clamp(timeline_scroll, 0, max_scroll)

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if solution:
                        playing   = not playing
                        last_step = time.time()
                if event.key == pygame.K_RIGHT and solution:
                    if step_idx < len(solution) - 1:
                        step_idx += 1
                if event.key == pygame.K_LEFT and solution:
                    if step_idx > 0:
                        step_idx -= 1
                if event.key == pygame.K_r:
                    puzzle   = random_puzzle()
                    solution = []; explored = []; stats = {}
                    step_idx = 0; playing = False; timeline_scroll = 0


if __name__ == "__main__":
    main()
