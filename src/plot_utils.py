from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


FONT_PATH = Path(r"C:\Windows\Fonts\arial.ttf")
BOLD_FONT_PATH = Path(r"C:\Windows\Fonts\arialbd.ttf")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(BOLD_FONT_PATH if bold else FONT_PATH), size)


def bar_chart(
    path: Path,
    title: str,
    groups: list[str],
    series: dict[str, list[float]],
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
    width: int = 1700,
    height: int = 950,
) -> None:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 220, 80, 120, 175
    chart_w = width - left - right
    chart_h = height - top - bottom
    draw.text((left, 30), title, fill="#17365D", font=font(36, True))

    for step in range(6):
        value = minimum + (maximum - minimum) * step / 5
        y = top + chart_h - chart_h * step / 5
        draw.line((left, y, left + chart_w, y), fill="#D9E2F3", width=2)
        draw.text(
            (80, y - 14),
            f"{value:.2f}",
            fill="#4A5568",
            font=font(24),
        )

    colors = ["#2E75B6", "#70AD47", "#ED7D31", "#A5A5A5", "#5B9BD5"]
    group_width = chart_w / max(len(groups), 1)
    bar_width = group_width / (len(series) + 1)
    for group_index, group in enumerate(groups):
        base_x = left + group_index * group_width
        for series_index, (series_name, values) in enumerate(series.items()):
            value = values[group_index]
            x0 = base_x + (series_index + 0.5) * bar_width
            x1 = x0 + bar_width * 0.82
            fraction = max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))
            y0 = top + chart_h * (1 - fraction)
            draw.rectangle(
                (x0, y0, x1, top + chart_h),
                fill=colors[series_index % len(colors)],
            )
            draw.text(
                (x0, y0 - 28),
                f"{value:.3f}",
                fill="#24364B",
                font=font(18),
            )
        label_box = draw.textbbox((0, 0), group, font=font(22))
        label_w = label_box[2] - label_box[0]
        draw.text(
            (base_x + group_width / 2 - label_w / 2, top + chart_h + 25),
            group,
            fill="#24364B",
            font=font(22),
        )

    legend_x = left
    legend_y = height - 70
    for index, series_name in enumerate(series):
        draw.rectangle(
            (legend_x, legend_y, legend_x + 24, legend_y + 24),
            fill=colors[index % len(colors)],
        )
        draw.text(
            (legend_x + 34, legend_y - 2),
            series_name,
            fill="#24364B",
            font=font(22),
        )
        legend_x += 260
    image.save(path, quality=95)


def line_chart(
    path: Path,
    title: str,
    x_values: list[int],
    series: dict[str, list[float]],
    *,
    minimum: float = 0.0,
    maximum: float = 1.0,
    width: int = 1700,
    height: int = 950,
) -> None:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    left, right, top, bottom = 170, 80, 120, 170
    chart_w = width - left - right
    chart_h = height - top - bottom
    draw.text((left, 30), title, fill="#17365D", font=font(36, True))
    for step in range(6):
        value = minimum + (maximum - minimum) * step / 5
        y = top + chart_h - chart_h * step / 5
        draw.line((left, y, left + chart_w, y), fill="#D9E2F3", width=2)
        draw.text((55, y - 14), f"{value:.2f}", fill="#4A5568", font=font(24))
    colors = ["#2E75B6", "#70AD47", "#ED7D31", "#A5A5A5"]
    x_min, x_max = min(x_values), max(x_values)
    for series_index, (name, values) in enumerate(series.items()):
        points = []
        for x, value in zip(x_values, values):
            px = left + chart_w * (x - x_min) / max(x_max - x_min, 1)
            py = top + chart_h * (
                1 - max(0.0, min(1.0, (value - minimum) / (maximum - minimum)))
            )
            points.append((px, py))
        draw.line(points, fill=colors[series_index], width=5)
    for x in [x_min, (x_min + x_max) // 2, x_max]:
        px = left + chart_w * (x - x_min) / max(x_max - x_min, 1)
        draw.text((px - 12, top + chart_h + 25), str(x), fill="#24364B", font=font(24))
    legend_x = left
    legend_y = height - 65
    for index, name in enumerate(series):
        draw.line(
            (legend_x, legend_y + 12, legend_x + 40, legend_y + 12),
            fill=colors[index],
            width=6,
        )
        draw.text((legend_x + 50, legend_y), name, fill="#24364B", font=font(22))
        legend_x += 380
    image.save(path, quality=95)


def confusion_chart(path: Path, title: str, matrix: list[list[int]]) -> None:
    image = Image.new("RGB", (950, 830), "white")
    draw = ImageDraw.Draw(image)
    draw.text((90, 35), title, fill="#17365D", font=font(34, True))
    origin_x, origin_y, cell = 270, 190, 250
    maximum = max(max(row) for row in matrix)
    for row in range(2):
        for col in range(2):
            value = matrix[row][col]
            intensity = int(245 - 155 * value / max(maximum, 1))
            fill = (intensity, intensity + 15 if intensity < 240 else 245, 255)
            x0, y0 = origin_x + col * cell, origin_y + row * cell
            draw.rectangle((x0, y0, x0 + cell, y0 + cell), fill=fill, outline="#6B86A3", width=3)
            text = str(value)
            box = draw.textbbox((0, 0), text, font=font(44, True))
            draw.text(
                (
                    x0 + cell / 2 - (box[2] - box[0]) / 2,
                    y0 + cell / 2 - (box[3] - box[1]) / 2,
                ),
                text,
                fill="#17365D",
                font=font(44, True),
            )
    labels_x = ["Predicted functional", "Predicted defect"]
    labels_y = ["Annotated functional", "Annotated defect"]
    for col, label in enumerate(labels_x):
        draw.text((origin_x + col * cell + 20, origin_y + cell * 2 + 25), label, fill="#24364B", font=font(22))
    for row, label in enumerate(labels_y):
        draw.text((25, origin_y + row * cell + 105), label, fill="#24364B", font=font(22))
    image.save(path, quality=95)
