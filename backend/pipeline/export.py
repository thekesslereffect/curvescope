import csv
import io


def export_tfop_report(analysis, event) -> dict:
    return {
        "tic_id": analysis.target.tic_id if analysis.target else "unknown",
        "common_name": analysis.target.common_name if analysis.target else None,
        "ra": analysis.target.ra if analysis.target else None,
        "dec": analysis.target.dec if analysis.target else None,
        "event_time_btjd": event.time_center,
        "duration_hours": event.duration_hours,
        "depth_ppm": event.depth_ppm,
        "anomaly_score": event.anomaly_score,
        "classifier_result": event.event_type,
        "notes": event.notes,
        "pipeline_version": "1.0",
        "detection_method": "1D convolutional autoencoder + BLS + wavelet + centroid",
    }


def events_to_csv(events: list[dict]) -> str:
    if not events:
        return ""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=events[0].keys())
    writer.writeheader()
    writer.writerows(events)
    return output.getvalue()
