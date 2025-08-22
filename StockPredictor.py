import csv
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from datetime import datetime

dates = []
prices = []

def get_data(csv_path):
    """Read date (MM/DD/YYYY) and price ($) columns into parallel lists."""
    with open(csv_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # skip header
        for row in reader:
            # date -> ordinal (int), price -> float (strip $ and ,)
            dates.append(datetime.strptime(row[0], "%m/%d/%Y").toordinal())
            prices.append(float(row[1].replace("$", "").replace(",", "")))


def _coerce_target_x(x, known_dates):
    """
    Accept several forms for x:
      - str "MM/DD/YYYY"  -> parse to ordinal
      - small int/float   -> treat as days after last known date (e.g., 29)
      - large int         -> assume it's already an ordinal date
    """
    if isinstance(x, str):
        return datetime.strptime(x, "%m/%d/%Y").toordinal()
    if isinstance(x, (int, float)):
        x_int = int(x)
        # Ordinals are usually around 730,000–740,000 these years.
        # If x looks "small", interpret it as an offset from the last date.
        if x_int < 50000:
            return max(known_dates) + x_int
        return x_int
    raise ValueError("Unsupported type for x. Use a date string, an offset int, or an ordinal int.")


def predict_price(dates, prices, x):
    """
    Fit SVR models and return predictions for target x.
    Returns (rbf_pred, lin_pred, poly_pred).
    """
    # Prepare feature matrix X and target y
    X_raw = np.array(dates, dtype=np.int64)
    X = (X_raw - X_raw.min()).reshape(-1, 1)  # scale: days since first date
    y = np.array(prices, dtype=float)

    # Models
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma='scale')

    # Fit
    svr_lin.fit(X, y)
    svr_poly.fit(X, y)
    svr_rbf.fit(X, y)

    # Plot
    plt.scatter(X, y, label='Data')
    plt.plot(X, svr_rbf.predict(X), label='RBF model')
    plt.plot(X, svr_lin.predict(X), label='Linear model')
    plt.plot(X, svr_poly.predict(X), label='Polynomial model')
    plt.xlabel('Days since first date')
    plt.ylabel('Price')
    plt.title('Stock Prices')
    plt.legend()
    plt.show()

    # Prepare target for prediction
    target_ord = _coerce_target_x(x, dates)
    x_arr = np.array([[target_ord - X_raw.min()]], dtype=np.int64)  # scale same as training

    return (
        float(svr_rbf.predict(x_arr)[0]),
        float(svr_lin.predict(x_arr)[0]),
        float(svr_poly.predict(x_arr)[0]),
    )


# --------- Run ----------
if __name__ == "__main__":
    get_data("HistoricalData_appl.csv")

    # Option A: keep your old behavior (e.g., "29" means 29 days after the last date)
    predicted_price = predict_price(dates, prices, 5)

    # Option B: predict a specific calendar date
    # predicted_price = predict_price(dates, prices, "08/01/2025")

    # Option C: pass an ordinal date directly
    # predicted_price = predict_price(dates, prices, datetime(2025, 8, 1).toordinal())
    from datetime import date, timedelta

    future_offset_days = 5  # this matches your call below
    last_date = date.fromordinal(max(dates))
    future_date = last_date + timedelta(days=future_offset_days)

    print(f"Predicted price on {future_date.strftime('%Y-%m-%d')} ({future_offset_days} days after last date):")

    rbf_pred, lin_pred, poly_pred = predicted_price
    print(f"  RBF:        ${rbf_pred:.2f}")
    print(f"  Linear:     ${lin_pred:.2f}")
    print(f"  Polynomial: ${poly_pred:.2f}")
