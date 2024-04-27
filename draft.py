tracks_amount_values = [5, 10, 20]
random_range_values = [10, 20]
bb_skip_percent_values = [0.5, 0.25]

# Начало Markdown таблицы
markdown_table = "| tracks_amount | random_range | bb_skip_percent | Metric |\n"
markdown_table += "|---------------|--------------|-----------------|--------|\n"

# Генерация строк таблицы
for tracks_amount in tracks_amount_values:
    for random_range in random_range_values:
        for bb_skip_percent in bb_skip_percent_values:
            markdown_table += f"| {tracks_amount} | {random_range} | {bb_skip_percent} | 0 |\n"

# Вывод таблицы
print(markdown_table)
