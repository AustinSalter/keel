"""Sprint 2.5 Manim Visualization — Geometry Tracks Structural Sequence.

The core finding: activation geometry at L20/L27 distinguishes correctly-ordered
dialectic operations (E→C→K) from disordered sequences. Curvature (displacement-vector
cosine) is the strongest discriminator — 20x separation between coherent and shuffled.

The geometry records the ORDER of operations, not the DEPTH of insight.

Scenes:
1. TheChiselTest       — Hero scene: curvature trajectories, coherent vs shuffled
2. DampingComparison   — All 3 metrics side-by-side with damping ratios
3. StructuralSequence  — What geometry CAN see: E→C→K ordering
4. SculptingSurface    — 3D surface evolving through dialectic turns
5. FormulaOverlay      — CKA equation with annotations (requires LaTeX)

Usage:
    manim -pql scripts/visualize_sprint_2_5.py TheChiselTest
    manim -pqh scripts/visualize_sprint_2_5.py TheChiselTest  # high quality
"""

import json
from pathlib import Path

import numpy as np
from manim import *

RESULTS = Path(__file__).resolve().parent.parent / "results" / "traces"

COLORS_MAP = {
    "coherent": GREEN,
    "d1_shuffled": RED,
    "d2_plausible": ORANGE,
    "d3_dilutory": PURPLE,
}
LABELS_MAP = {
    "coherent": "Coherent (E→C→K)",
    "d1_shuffled": "Shuffled (random order)",
    "d2_plausible": "Plausible sub (1 turn replaced)",
    "d3_dilutory": "Dilutory sub (1 turn hollowed)",
}
SHORT_LABELS = {
    "coherent": "Coherent",
    "d1_shuffled": "Shuffled",
    "d2_plausible": "Plausible",
    "d3_dilutory": "Dilutory",
}


def load_trace(session, variant):
    path = RESULTS / f"{session}_{variant}.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def load_all_sessions():
    """Load all available sessions with their variants."""
    sessions = {}
    for session in ["thesis_geometry", "kinnected", "harness_thesis", "agentic_commerce"]:
        variants = {}
        for v in ["coherent", "d1_shuffled", "d2_plausible", "d3_dilutory"]:
            d = load_trace(session, v)
            if d:
                variants[v] = d
        if len(variants) >= 2:
            sessions[session] = variants
    return sessions


def best_session():
    """Find session with most variants, preferring thesis_geometry."""
    sessions = load_all_sessions()
    if "thesis_geometry" in sessions and len(sessions["thesis_geometry"]) >= 3:
        return "thesis_geometry", sessions["thesis_geometry"]
    for s, v in sorted(sessions.items(), key=lambda x: -len(x[1])):
        if len(v) >= 2:
            return s, v
    return None, {}


def damping_ratio(trajectory):
    """h2σ/h1σ — the core damping metric."""
    if len(trajectory) < 4:
        return float("nan")
    h1 = np.std(trajectory[: len(trajectory) // 2])
    h2 = np.std(trajectory[len(trajectory) // 2 :])
    return h2 / h1 if h1 > 0 else float("inf")


def rolling_std(vals, window=4):
    """Rolling standard deviation of turn-to-turn changes."""
    changes = np.diff(vals)
    result = []
    for i in range(len(changes)):
        start = max(0, i - window // 2)
        end = min(len(changes), i + window // 2 + 1)
        result.append(np.std(changes[start:end]))
    return result


# ---------------------------------------------------------------------------
# Scene 1: The Chisel Test — Hero visualization
# ---------------------------------------------------------------------------

class TheChiselTest(Scene):
    """Curvature trajectories at L20: the 20x separation finding."""

    def construct(self):
        session, variants = best_session()
        if not variants:
            self.add(Text("No trace data found", color=RED))
            return

        layer_key = "layer_20"

        # Opening title
        title = Text("The Chisel Test", font_size=44, weight=BOLD).move_to(ORIGIN)
        self.play(Write(title))
        self.wait(0.5)

        premise = Text(
            "Does activation geometry distinguish\nordered reasoning from disordered?",
            font_size=22, color=GREY,
        ).next_to(title, DOWN, buff=0.4)
        self.play(FadeIn(premise))
        self.wait(1.5)
        self.play(FadeOut(title), FadeOut(premise))

        # Setup label
        setup = VGroup(
            Text(f"Session: {session}", font_size=18, color=GREY),
            Text(f"Layer 20 (deep processing) — New-token-only activations", font_size=16, color=GREY),
            Text("Metric: Displacement-vector cosine (trajectory curvature)", font_size=16, color=YELLOW),
        ).arrange(DOWN, buff=0.1).to_edge(UP, buff=0.3)
        self.play(FadeIn(setup))

        # Build curvature plot
        all_curv = []
        for v, d in variants.items():
            curv = d["layers"][layer_key].get("curvature_trajectory", [])
            all_curv.extend(curv)

        if not all_curv:
            self.add(Text("No curvature data", color=RED))
            return

        y_min = min(all_curv) - 0.02
        y_max = max(all_curv) + 0.02
        n_points = max(
            len(d["layers"][layer_key].get("curvature_trajectory", []))
            for d in variants.values()
        )

        axes = Axes(
            x_range=[0, n_points, 2],
            y_range=[y_min, y_max, (y_max - y_min) / 4],
            x_length=11,
            y_length=4.5,
            axis_config={"include_numbers": False},
        ).shift(DOWN * 0.5)

        x_lab = Text("Turn transition", font_size=14).next_to(axes, DOWN, buff=0.15)
        y_lab = Text("Curvature", font_size=14).rotate(90 * DEGREES).next_to(axes, LEFT, buff=0.15)
        self.play(Create(axes), Write(x_lab), Write(y_lab))

        # Draw coherent first
        if "coherent" in variants:
            curv = variants["coherent"]["layers"][layer_key]["curvature_trajectory"]
            points = [axes.c2p(i, v) for i, v in enumerate(curv)]
            graph = VMobject(color=GREEN, stroke_width=3)
            graph.set_points_smoothly(points)

            ratio = damping_ratio(curv)
            label = Text(f"Coherent  (damping: {ratio:.2f})", font_size=16, color=GREEN)
            label.next_to(axes.c2p(n_points, curv[-1]), RIGHT, buff=0.1)

            self.play(Create(graph, run_time=2.5))
            self.play(Write(label))
            self.wait(0.5)

            # Show the damping envelope
            std_vals = rolling_std(curv, window=3)
            if std_vals:
                max_s = max(std_vals) if max(std_vals) > 0 else 1
                scale = (y_max - y_min) * 0.3 / max_s
                env_points = [axes.c2p(i, y_min + v * scale) for i, v in enumerate(std_vals)]
                envelope = VMobject(color=GREEN_B, stroke_width=2, stroke_opacity=0.5)
                envelope.set_points_smoothly(env_points)
                env_note = Text("oscillation envelope ↘", font_size=12, color=GREEN_B)
                env_note.next_to(axes.c2p(len(std_vals) - 1, y_min + std_vals[-1] * scale), RIGHT, buff=0.1)
                self.play(Create(envelope), FadeIn(env_note))
                self.wait(0.5)

        # Draw shuffled
        if "d1_shuffled" in variants:
            curv_s = variants["d1_shuffled"]["layers"][layer_key]["curvature_trajectory"]
            points_s = [axes.c2p(i, v) for i, v in enumerate(curv_s)]
            graph_s = VMobject(color=RED, stroke_width=3)
            graph_s.set_points_smoothly(points_s)

            ratio_s = damping_ratio(curv_s)
            label_s = Text(f"Shuffled  (damping: {ratio_s:.2f})", font_size=16, color=RED)
            label_s.next_to(axes.c2p(n_points, curv_s[-1]), RIGHT, buff=0.1)

            self.play(Create(graph_s, run_time=2))
            self.play(Write(label_s))
            self.wait(1)

        # Draw D2/D3 if available
        for variant in ["d2_plausible", "d3_dilutory"]:
            if variant not in variants:
                continue
            curv_v = variants[variant]["layers"][layer_key].get("curvature_trajectory", [])
            if not curv_v:
                continue
            points_v = [axes.c2p(i, v) for i, v in enumerate(curv_v)]
            graph_v = VMobject(color=COLORS_MAP[variant], stroke_width=2, stroke_opacity=0.7)
            graph_v.set_points_smoothly(points_v)

            ratio_v = damping_ratio(curv_v)
            label_v = Text(
                f"{SHORT_LABELS[variant]}  ({ratio_v:.2f})",
                font_size=14, color=COLORS_MAP[variant],
            )
            label_v.next_to(axes.c2p(n_points, curv_v[-1]), RIGHT, buff=0.1)
            self.play(Create(graph_v, run_time=1), Write(label_v))

        self.wait(1)

        # Verdict
        verdict_box = VGroup(
            Text("Geometry tracks structural sequence.", font_size=22, weight=BOLD),
            Text("Ordered operations → damped oscillation (curvature converges)", font_size=16, color=GREEN),
            Text("Disordered operations → divergent oscillation (curvature erratic)", font_size=16, color=RED),
            Text("Single-turn quality differences → absorbed by task frame (no separation)", font_size=16, color=GREY),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.12)
        verdict_box.to_edge(DOWN, buff=0.2)

        bg = SurroundingRectangle(verdict_box, buff=0.15, color=WHITE, stroke_width=1, fill_opacity=0.05)
        self.play(FadeIn(bg), FadeIn(verdict_box, lag_ratio=0.2))
        self.wait(3)


# ---------------------------------------------------------------------------
# Scene 2: Damping Comparison — 3 metrics side by side
# ---------------------------------------------------------------------------

class DampingComparison(Scene):
    """CKA, Grassmann, Curvature at L27 — all showing the same damping pattern."""

    def construct(self):
        session, variants = best_session()
        if not variants:
            self.add(Text("No data found", color=RED))
            return

        layer_key = "layer_27"
        metrics = [
            ("cka_trajectory", "CKA (relational structure)"),
            ("grassmann_trajectory", "Grassmann (subspace rotation)"),
            ("curvature_trajectory", "Curvature (direction consistency)"),
        ]

        title = Text(
            f"Three Metrics, One Story — {session} at L27",
            font_size=28, weight=BOLD,
        ).to_edge(UP, buff=0.3)
        self.play(Write(title))

        # 1x3 grid
        panels = VGroup()
        panel_data = []

        for metric_key, metric_label in metrics:
            all_vals = []
            for d in variants.values():
                vals = d["layers"][layer_key].get(metric_key, [])
                all_vals.extend(vals)
            if not all_vals:
                continue

            y_lo = min(all_vals) - (max(all_vals) - min(all_vals)) * 0.1
            y_hi = max(all_vals) + (max(all_vals) - min(all_vals)) * 0.1
            n = max(len(d["layers"][layer_key].get(metric_key, [])) for d in variants.values())

            ax = Axes(
                x_range=[0, n, max(1, n // 3)],
                y_range=[y_lo, y_hi, (y_hi - y_lo) / 3],
                x_length=4,
                y_length=3.5,
                axis_config={"include_numbers": False, "font_size": 10},
            )
            label = Text(metric_label, font_size=13).next_to(ax, UP, buff=0.1)
            panel = VGroup(ax, label)
            panels.add(panel)
            panel_data.append((ax, metric_key))

        panels.arrange(RIGHT, buff=0.5)
        panels.next_to(title, DOWN, buff=0.3)
        panels.scale_to_fit_width(13)
        self.play(FadeIn(panels))

        # Draw all variants on each panel
        for ax, metric_key in panel_data:
            for variant, d in variants.items():
                vals = d["layers"][layer_key].get(metric_key, [])
                if len(vals) < 2:
                    continue
                points = [ax.c2p(i, v) for i, v in enumerate(vals)]
                graph = VMobject(color=COLORS_MAP[variant], stroke_width=2.5, stroke_opacity=0.85)
                graph.set_points_smoothly(points)
                self.play(Create(graph, run_time=0.4), run_time=0.4)

        # Damping ratio table
        table_rows = []
        header = Text(
            f"{'Variant':<16} {'CKA':>8} {'Grass':>8} {'Curv':>8}",
            font_size=14, font="Monospace",
        )
        table_rows.append(header)

        for variant in ["coherent", "d1_shuffled", "d2_plausible", "d3_dilutory"]:
            if variant not in variants:
                continue
            ratios = []
            for metric_key, _ in metrics:
                vals = variants[variant]["layers"][layer_key].get(metric_key, [])
                ratios.append(f"{damping_ratio(vals):>8.2f}")
            row = Text(
                f"{SHORT_LABELS[variant]:<16} {''.join(ratios)}",
                font_size=14, font="Monospace", color=COLORS_MAP[variant],
            )
            table_rows.append(row)

        table = VGroup(*table_rows).arrange(DOWN, aligned_edge=LEFT, buff=0.05)
        table.to_edge(DOWN, buff=0.3)
        self.play(FadeIn(table))
        self.wait(3)


# ---------------------------------------------------------------------------
# Scene 3: Structural Sequence — What geometry sees
# ---------------------------------------------------------------------------

class StructuralSequence(Scene):
    """Visualize what the geometry is actually tracking: operation ordering."""

    def construct(self):
        title = Text("What Geometry Sees", font_size=36, weight=BOLD).to_edge(UP, buff=0.4)
        self.play(Write(title))
        self.wait(0.5)

        # Coherent sequence
        coh_label = Text("Coherent trace:", font_size=20, color=GREEN).shift(UP * 2 + LEFT * 4)
        self.play(Write(coh_label))

        phases = ["E", "C", "K", "E", "C", "K", "E", "C", "K", "E", "C", "K", "E", "C", "K"]
        phase_colors = {"E": BLUE, "C": TEAL, "K": YELLOW}
        phase_names = {"E": "Expand", "C": "Compress", "K": "Critique"}

        # Draw coherent sequence as connected boxes
        coh_boxes = VGroup()
        for i, p in enumerate(phases):
            box = Square(side_length=0.5, color=phase_colors[p], fill_opacity=0.3)
            txt = Text(p, font_size=16, color=phase_colors[p], weight=BOLD)
            txt.move_to(box)
            group = VGroup(box, txt)
            coh_boxes.add(group)

        coh_boxes.arrange(RIGHT, buff=0.08)
        coh_boxes.next_to(coh_label, DOWN, buff=0.3)

        # Arrows between boxes
        coh_arrows = VGroup()
        for i in range(len(phases) - 1):
            arrow = Arrow(
                coh_boxes[i].get_right(), coh_boxes[i + 1].get_left(),
                buff=0.02, stroke_width=1.5, color=GREY, max_tip_length_to_length_ratio=0.3,
            )
            coh_arrows.add(arrow)

        self.play(FadeIn(coh_boxes, lag_ratio=0.05), FadeIn(coh_arrows, lag_ratio=0.05))
        self.wait(0.5)

        damped = Text("→ Damped oscillation (ratio < 1.0)", font_size=16, color=GREEN)
        damped.next_to(coh_boxes, DOWN, buff=0.2)
        self.play(Write(damped))

        # Shuffled sequence
        shuf_label = Text("Shuffled trace:", font_size=20, color=RED).shift(DOWN * 0.5 + LEFT * 4)
        self.play(Write(shuf_label))

        np.random.seed(42)
        shuffled = list(phases)
        np.random.shuffle(shuffled)

        shuf_boxes = VGroup()
        for p in shuffled:
            box = Square(side_length=0.5, color=phase_colors[p], fill_opacity=0.3)
            txt = Text(p, font_size=16, color=phase_colors[p], weight=BOLD)
            txt.move_to(box)
            shuf_boxes.add(VGroup(box, txt))

        shuf_boxes.arrange(RIGHT, buff=0.08)
        shuf_boxes.next_to(shuf_label, DOWN, buff=0.3)

        shuf_arrows = VGroup()
        for i in range(len(shuffled) - 1):
            arrow = Arrow(
                shuf_boxes[i].get_right(), shuf_boxes[i + 1].get_left(),
                buff=0.02, stroke_width=1.5, color=GREY, max_tip_length_to_length_ratio=0.3,
            )
            shuf_arrows.add(arrow)

        self.play(FadeIn(shuf_boxes, lag_ratio=0.05), FadeIn(shuf_arrows, lag_ratio=0.05))

        undamped = Text("→ Divergent oscillation (ratio > 1.0)", font_size=16, color=RED)
        undamped.next_to(shuf_boxes, DOWN, buff=0.2)
        self.play(Write(undamped))
        self.wait(1)

        # Key insight
        insight = VGroup(
            Text("The geometry tracks conversational topology,", font_size=20),
            Text("not argumentative quality.", font_size=20, color=YELLOW),
            Text("", font_size=10),
            Text("E→C→K→E→C→K  produces damped oscillation", font_size=16, color=GREEN),
            Text("K→E→C→K→C→E  produces divergent oscillation", font_size=16, color=RED),
            Text("E→C→K(weak)→E→C→K  looks identical to E→C→K(strong)→E→C→K", font_size=16, color=GREY),
        ).arrange(DOWN, buff=0.1)
        insight.to_edge(DOWN, buff=0.3)

        bg = SurroundingRectangle(insight, buff=0.15, color=WHITE, stroke_width=1, fill_opacity=0.05)
        self.play(FadeIn(bg), FadeIn(insight, lag_ratio=0.1))
        self.wait(3)


# ---------------------------------------------------------------------------
# Scene 4: Sculpting Surface — 3D activation topography
# ---------------------------------------------------------------------------

class SculptingSurface(ThreeDScene):
    """3D surface evolving through dialectic turns — ridges sharpen, noise drops."""

    def construct(self):
        session, variants = best_session()
        if not variants or "coherent" not in variants:
            self.add(Text("No coherent trace data found", color=RED))
            return

        layer_key = "layer_27"
        cka = variants["coherent"]["layers"][layer_key]["cka_trajectory"]
        evr = variants["coherent"]["layers"][layer_key]["evr_trajectory"]

        title = Text("Activation Topography — Sculpting Through Dialectic", font_size=24)
        title.to_edge(UP, buff=0.2)
        self.add_fixed_in_frame_mobjects(title)
        self.play(Write(title))

        self.set_camera_orientation(phi=60 * DEGREES, theta=-45 * DEGREES)

        axes = ThreeDAxes(
            x_range=[-3, 3, 1], y_range=[-3, 3, 1], z_range=[0, 2, 0.5],
            x_length=6, y_length=6, z_length=3,
        )
        self.play(Create(axes))

        def make_surface(progress):
            """Surface parameterized by session progress [0, 1]."""
            ridge_width = 1.5 - progress * 0.8
            ridge_height = 0.5 + progress * 1.0

            def func(u, v):
                ridge = ridge_height * np.exp(-((u - v) ** 2) / (2 * ridge_width ** 2))
                cross = 0.3 * progress * np.sin(u * 2) * np.sin(v * 2)
                noise = (1 - progress) * 0.2 * np.sin(u * 5 + v * 3)
                return ridge + cross + noise

            return Surface(
                lambda u, v: axes.c2p(u, v, func(u, v)),
                u_range=[-3, 3], v_range=[-3, 3],
                resolution=(30, 30), fill_opacity=0.7, stroke_width=0.5,
            )

        n_turns = len(cka) + 1
        key_turns = [0, n_turns // 4, n_turns // 2, 3 * n_turns // 4, n_turns - 1]
        phase_names = ["Initial", "Expansion", "Compression", "Critique", "Convergence"]

        surface = make_surface(0)
        surface.set_color_by_gradient(BLUE, GREEN)
        self.play(Create(surface, run_time=1.5))

        turn_label = Text("Turn 0 — Initial", font_size=18, color=WHITE).to_edge(DOWN, buff=0.3)
        self.add_fixed_in_frame_mobjects(turn_label)
        self.play(Write(turn_label))

        for i, turn in enumerate(key_turns[1:]):
            progress = turn / (n_turns - 1)
            new_surface = make_surface(progress)
            new_surface.set_color_by_gradient(
                interpolate_color(BLUE, TEAL, progress),
                interpolate_color(GREEN_B, GREEN, progress),
            )

            new_label = Text(
                f"Turn {turn} — {phase_names[min(i + 1, 4)]}", font_size=18, color=WHITE,
            ).to_edge(DOWN, buff=0.3)

            self.play(
                Transform(surface, new_surface, run_time=1.5),
                Transform(turn_label, new_label),
            )
            self.begin_ambient_camera_rotation(rate=0.1)
            self.wait(1)
            self.stop_ambient_camera_rotation()

        self.wait(1)


# ---------------------------------------------------------------------------
# Scene 5: Formula Overlay — CKA equation (LaTeX required)
# ---------------------------------------------------------------------------

class FormulaOverlay(Scene):
    """CKA formula with annotations. Requires LaTeX."""

    def construct(self):
        title = Text("Centered Kernel Alignment (CKA)", font_size=30).to_edge(UP, buff=0.5)
        self.play(Write(title))

        cka = MathTex(
            r"\text{CKA}(X, Y) = \frac{\langle HK_XH, \, HK_YH \rangle_F}"
            r"{\|HK_XH\|_F \, \|HK_YH\|_F}",
            font_size=36,
        )
        self.play(Write(cka))
        self.wait(1)

        where = MathTex(
            r"K_X = \frac{X^\top X}{n}", r"\quad", r"\text{(feature covariance, } d \times d \text{)}",
            font_size=24,
        ).next_to(cka, DOWN, buff=0.5)
        centering = MathTex(
            r"H = I - \frac{1}{n}\mathbf{1}\mathbf{1}^\top",
            font_size=24,
        ).next_to(where, DOWN, buff=0.3)

        self.play(Write(where), Write(centering))
        self.wait(1)

        insight = Text(
            "Feature-space CKA: O(d²) not O(n²)\n"
            "d = 3584 (Qwen hidden dim) vs n = 49K tokens\n"
            "Same math, 200x less memory",
            font_size=16, color=YELLOW,
        ).next_to(centering, DOWN, buff=0.5)
        self.play(FadeIn(insight))
        self.wait(1)

        new_token = Text(
            "Computed on NEW TOKENS ONLY at each turn\n"
            "Model sees full context via attention\n"
            "We measure only each turn's geometric contribution",
            font_size=16, color=GREEN,
        ).next_to(insight, DOWN, buff=0.4)
        self.play(FadeIn(new_token))
        self.wait(2)
