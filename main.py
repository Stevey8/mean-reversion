from utils.dataloader import get_sp500, select_global_candidates
from production import update_data_pkl, update_models_pkl, check_backtest, run_pred
from config import watchlist

import os
import sys
import functools
import warnings
from zoneinfo import ZoneInfo
import schedule
from datetime import time as dt_time, datetime, timedelta
import time  # for sleep()

warnings.filterwarnings("ignore")

TZ = ZoneInfo("America/Toronto")
MARKET_CLOSE = dt_time(16, 30)  # 4:30pm ET
MARKET_OPEN  = dt_time(9, 30)   # 9:30am ET
LOCKFILE = "/tmp/weekly_retraining.lock"

def within_weekend_retrain_window(now: datetime) -> bool:
    """
    True if we're in the intended window:
      - Fri after market close
      - Any time Sat/Sun
      - Mon before market open
    """
    now_et = now.astimezone(TZ)
    wd = now_et.weekday()  # Mon=0..Sun=6
    if wd == 4:  # Fri
        return now_et.time() >= MARKET_CLOSE
    if wd in (5, 6):  # Sat/Sun
        return True
    if wd == 0:  # Mon
        return now_et.time() < MARKET_OPEN
    return False

def single_run_lock(path=LOCKFILE):
    """
    Decorator to ensure only one run at a time via a simple lockfile.
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
            except FileExistsError:
                print("[weekly_retraining] Another run is in progress. Skipping.")
                return
            try:
                return fn(*args, **kwargs)
            finally:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
        return wrapper
    return decorator

@single_run_lock()
def weekly_retraining():
    now = datetime.now(tz=TZ)
    if not within_weekend_retrain_window(now):
        print(f"[weekly_retraining] Not in retrain window at {now}. Skipping.")
        return
    print("=== Running Weekly Retraining ===")
    update_models_pkl()
    print("Retraining complete. Models updated and saved.")

def check_backtest_job():
    now = datetime.now(tz=TZ)
    print(f"[check_backtest] {now.isoformat(timespec='seconds')}")
    check_backtest()

def setup_schedule():
    # retrain after close on Fridays
    schedule.every().friday.at("17:55").do(weekly_retraining)
    # backtest checks Sun–Thu after close-ish
    for day in ("sunday", "monday", "tuesday", "wednesday", "thursday"):
        getattr(schedule.every(), day).at("17:55").do(check_backtest_job)

def run_forever():
    setup_schedule()
    print("[scheduler] started")
    while True:
        schedule.run_pending()
        time.sleep(1)

def main():
    print("Starting scheduler. Press Ctrl+C to stop.\n")
    print(
        "Available names:\n"
        "  watchlist\n"
        "  get_sp500(), select_global_candidates()\n"
        "  update_data_pkl(), update_models_pkl(), check_backtest(), run_pred()\n"
        "  setup_schedule(), weekly_retraining(), check_backtest_job(), run_forever()\n"
        "  TZ, MARKET_OPEN, MARKET_CLOSE\n"
    )
    # If launched with `python -i`, don't start the loop automatically
    if hasattr(sys, 'ps1') or sys.flags.interactive:
        print("Interactive shell detected. Scheduler not started automatically.")
    else:
        run_forever()

if __name__ == '__main__':
    main()
