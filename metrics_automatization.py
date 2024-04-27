import importlib
import csv
from metrics import CustomTrackingMetric
from models import EasyModel

if __name__ == '__main__':
    tracks_amount_values = [5, 10, 20]
    random_range_values = [10, 20]
    bb_skip_percent_values = [0.5, 0.25]

    easy_model = EasyModel()
    metric_model = CustomTrackingMetric()

    with open('metrics_results.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['tracks_amount', 'random_range', 'bb_skip_percent', 'average_match_ratio'])

        for tracks_amount in tracks_amount_values:
            for random_range in random_range_values:
                for bb_skip_percent in bb_skip_percent_values:
                    module_name = f"track_{tracks_amount}_{random_range}_{str(bb_skip_percent).replace('.', '')}"
                    track_module = importlib.import_module(module_name)

                    track_data = track_module.track_data
                    for el in track_data:
                        predict = easy_model.predict(el)
                        metric_model.add_row(predict)

                    average_match_ratio = metric_model.calculate_metric()
                    print(f"Metrics for {tracks_amount}_{random_range}_{bb_skip_percent}: {average_match_ratio}")
                    writer.writerow([tracks_amount, random_range, bb_skip_percent, average_match_ratio])

                    metric_model = CustomTrackingMetric()  # Reset the metric model for the next file
