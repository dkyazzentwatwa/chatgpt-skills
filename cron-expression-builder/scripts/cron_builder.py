#!/usr/bin/env python3
"""
Cron Expression Builder - Build and validate cron expressions.
"""

import argparse
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

try:
    from croniter import croniter
    CRONITER_AVAILABLE = True
except ImportError:
    CRONITER_AVAILABLE = False


class CronBuilder:
    """Build, parse, and validate cron expressions."""

    PRESETS = {
        "every_minute": "* * * * *",
        "every_5_minutes": "*/5 * * * *",
        "every_10_minutes": "*/10 * * * *",
        "every_15_minutes": "*/15 * * * *",
        "every_30_minutes": "*/30 * * * *",
        "hourly": "0 * * * *",
        "every_2_hours": "0 */2 * * *",
        "every_6_hours": "0 */6 * * *",
        "daily_midnight": "0 0 * * *",
        "daily_noon": "0 12 * * *",
        "daily_6am": "0 6 * * *",
        "daily_6pm": "0 18 * * *",
        "weekly_sunday": "0 0 * * 0",
        "weekly_monday": "0 0 * * 1",
        "weekdays_9am": "0 9 * * 1-5",
        "monthly": "0 0 1 * *",
        "monthly_15th": "0 0 15 * *",
        "quarterly": "0 0 1 1,4,7,10 *",
        "yearly": "0 0 1 1 *"
    }

    WEEKDAYS = {
        "sunday": 0, "sun": 0,
        "monday": 1, "mon": 1,
        "tuesday": 2, "tue": 2,
        "wednesday": 3, "wed": 3,
        "thursday": 4, "thu": 4,
        "friday": 5, "fri": 5,
        "saturday": 6, "sat": 6
    }

    MONTHS = {
        "january": 1, "jan": 1,
        "february": 2, "feb": 2,
        "march": 3, "mar": 3,
        "april": 4, "apr": 4,
        "may": 5,
        "june": 6, "jun": 6,
        "july": 7, "jul": 7,
        "august": 8, "aug": 8,
        "september": 9, "sep": 9,
        "october": 10, "oct": 10,
        "november": 11, "nov": 11,
        "december": 12, "dec": 12
    }

    def __init__(self, with_seconds: bool = False):
        """
        Initialize the cron builder.

        Args:
            with_seconds: Use 6-field format with seconds
        """
        self.with_seconds = with_seconds

    def from_text(self, text: str) -> str:
        """
        Convert natural language to cron expression.

        Args:
            text: Natural language description

        Returns:
            Cron expression
        """
        text = text.lower().strip()

        # Default values
        minute = "*"
        hour = "*"
        day = "*"
        month = "*"
        weekday = "*"

        # Every X minutes
        match = re.search(r'every\s+(\d+)\s*min', text)
        if match:
            interval = int(match.group(1))
            minute = f"*/{interval}"
            return self.build(minute=minute, hour="*", day="*", month="*", weekday="*")

        # Every minute
        if re.search(r'every\s+min', text):
            return self.build(minute="*", hour="*", day="*", month="*", weekday="*")

        # Every X hours
        match = re.search(r'every\s+(\d+)\s*hour', text)
        if match:
            interval = int(match.group(1))
            return self.build(minute="0", hour=f"*/{interval}", day="*", month="*", weekday="*")

        # Every hour / hourly
        if re.search(r'every\s+hour|hourly', text):
            return self.build(minute="0", hour="*", day="*", month="*", weekday="*")

        # Parse time (HH:MM AM/PM or HH:MM or noon/midnight)
        time_match = re.search(r'(\d{1,2}):(\d{2})\s*(am|pm)?', text, re.IGNORECASE)
        noon_match = re.search(r'\bnoon\b', text)
        midnight_match = re.search(r'\bmidnight\b', text)

        if time_match:
            h = int(time_match.group(1))
            m = int(time_match.group(2))
            ampm = time_match.group(3)

            if ampm:
                if ampm.lower() == 'pm' and h != 12:
                    h += 12
                elif ampm.lower() == 'am' and h == 12:
                    h = 0

            hour = str(h)
            minute = str(m)
        elif noon_match:
            hour = "12"
            minute = "0"
        elif midnight_match:
            hour = "0"
            minute = "0"
        else:
            # Check for just hour (e.g., "at 9am", "at 3pm")
            hour_match = re.search(r'(?:at\s+)?(\d{1,2})\s*(am|pm)', text, re.IGNORECASE)
            if hour_match:
                h = int(hour_match.group(1))
                ampm = hour_match.group(2)

                if ampm.lower() == 'pm' and h != 12:
                    h += 12
                elif ampm.lower() == 'am' and h == 12:
                    h = 0

                hour = str(h)
                minute = "0"

        # Check for day of week
        for day_name, day_num in self.WEEKDAYS.items():
            if day_name in text:
                weekday = str(day_num)
                break

        # Check for weekdays/weekends
        if 'weekday' in text:
            weekday = "1-5"
        elif 'weekend' in text:
            weekday = "0,6"

        # Check for day of month
        day_match = re.search(r'(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?(?:every\s+)?month', text)
        if day_match:
            day = day_match.group(1)
        elif 'first' in text and 'month' in text:
            day = "1"
        elif 'last' in text and 'month' in text:
            day = "L"

        # Check for specific month
        for month_name, month_num in self.MONTHS.items():
            if month_name in text:
                month = str(month_num)
                break

        # Daily
        if re.search(r'\bdaily\b|every\s+day', text):
            weekday = "*"
            day = "*"

        return self.build(minute=minute, hour=hour, day=day, month=month, weekday=weekday)

    def build(self, minute: str = "*", hour: str = "*",
             day: str = "*", month: str = "*",
             weekday: str = "*") -> str:
        """
        Build cron expression from fields.

        Args:
            minute: Minute field (0-59)
            hour: Hour field (0-23)
            day: Day of month field (1-31)
            month: Month field (1-12)
            weekday: Day of week field (0-6)

        Returns:
            Cron expression
        """
        if self.with_seconds:
            return f"0 {minute} {hour} {day} {month} {weekday}"
        return f"{minute} {hour} {day} {month} {weekday}"

    def describe(self, expression: str) -> str:
        """
        Convert cron expression to human-readable description.

        Args:
            expression: Cron expression

        Returns:
            Human-readable description
        """
        parts = expression.split()
        if len(parts) == 6:
            # Remove seconds field for parsing
            parts = parts[1:]
        elif len(parts) != 5:
            return "Invalid cron expression"

        minute, hour, day, month, weekday = parts

        descriptions = []

        # Time description
        if minute == "*" and hour == "*":
            descriptions.append("Every minute")
        elif minute.startswith("*/"):
            interval = minute[2:]
            if hour == "*":
                descriptions.append(f"Every {interval} minutes")
            else:
                descriptions.append(f"Every {interval} minutes during hour {hour}")
        elif hour.startswith("*/"):
            interval = hour[2:]
            descriptions.append(f"Every {interval} hours")
        elif minute != "*" and hour != "*":
            h = int(hour)
            m = int(minute)
            ampm = "AM" if h < 12 else "PM"
            h_display = h if h <= 12 else h - 12
            if h_display == 0:
                h_display = 12
            descriptions.append(f"At {h_display}:{m:02d} {ampm}")
        elif hour != "*":
            h = int(hour)
            ampm = "AM" if h < 12 else "PM"
            h_display = h if h <= 12 else h - 12
            if h_display == 0:
                h_display = 12
            descriptions.append(f"At {h_display}:00 {ampm}")

        # Day of week
        weekday_names = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        if weekday != "*":
            if weekday == "1-5":
                descriptions.append("Monday through Friday")
            elif weekday == "0,6":
                descriptions.append("weekends")
            elif "-" in weekday:
                start, end = weekday.split("-")
                descriptions.append(f"{weekday_names[int(start)]} through {weekday_names[int(end)]}")
            elif "," in weekday:
                days = [weekday_names[int(d)] for d in weekday.split(",")]
                descriptions.append(", ".join(days))
            else:
                descriptions.append(f"only on {weekday_names[int(weekday)]}")

        # Day of month
        if day != "*":
            if day == "L":
                descriptions.append("on the last day of the month")
            else:
                descriptions.append(f"on day {day} of the month")

        # Month
        month_names = ["", "January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"]
        if month != "*":
            if "," in month:
                months = [month_names[int(m)] for m in month.split(",")]
                descriptions.append(f"in {', '.join(months)}")
            else:
                descriptions.append(f"in {month_names[int(month)]}")

        return ", ".join(descriptions) if descriptions else "Every minute"

    def parse(self, expression: str) -> Dict:
        """
        Parse cron expression into components.

        Args:
            expression: Cron expression

        Returns:
            Dictionary with field values
        """
        parts = expression.split()

        if len(parts) == 6:
            return {
                "second": parts[0],
                "minute": parts[1],
                "hour": parts[2],
                "day_of_month": parts[3],
                "month": parts[4],
                "day_of_week": parts[5],
                "format": "6-field"
            }
        elif len(parts) == 5:
            return {
                "minute": parts[0],
                "hour": parts[1],
                "day_of_month": parts[2],
                "month": parts[3],
                "day_of_week": parts[4],
                "format": "5-field"
            }
        else:
            return {"error": "Invalid cron expression format"}

    def validate(self, expression: str) -> Dict:
        """
        Validate cron expression.

        Args:
            expression: Cron expression

        Returns:
            Validation result
        """
        parsed = self.parse(expression)

        if "error" in parsed:
            return {"valid": False, "error": parsed["error"]}

        # Define valid ranges
        ranges = {
            "second": (0, 59),
            "minute": (0, 59),
            "hour": (0, 23),
            "day_of_month": (1, 31),
            "month": (1, 12),
            "day_of_week": (0, 6)
        }

        for field, value in parsed.items():
            if field in ["format", "error"]:
                continue

            if field not in ranges:
                continue

            min_val, max_val = ranges[field]

            if value == "*":
                continue

            # Handle */n
            if value.startswith("*/"):
                try:
                    n = int(value[2:])
                    if n < 1:
                        return {"valid": False, "error": f"Invalid interval in {field}: {value}"}
                except ValueError:
                    return {"valid": False, "error": f"Invalid {field}: {value}"}
                continue

            # Handle ranges (n-m)
            if "-" in value and not value.startswith("-"):
                try:
                    start, end = value.split("-")
                    s, e = int(start), int(end)
                    if s < min_val or e > max_val or s > e:
                        return {"valid": False, "error": f"Invalid range in {field}: {value}"}
                except ValueError:
                    return {"valid": False, "error": f"Invalid {field}: {value}"}
                continue

            # Handle lists (n,m,o)
            if "," in value:
                try:
                    for v in value.split(","):
                        n = int(v)
                        if n < min_val or n > max_val:
                            return {"valid": False, "error": f"Invalid value in {field}: {v}"}
                except ValueError:
                    return {"valid": False, "error": f"Invalid {field}: {value}"}
                continue

            # Handle special characters
            if value in ["L", "W", "?"]:
                continue

            # Single value
            try:
                n = int(value)
                if n < min_val or n > max_val:
                    return {"valid": False, "error": f"Invalid {field}: {n} (must be {min_val}-{max_val})"}
            except ValueError:
                return {"valid": False, "error": f"Invalid {field}: {value}"}

        return {
            "valid": True,
            "expression": expression,
            "fields": parsed,
            "description": self.describe(expression)
        }

    def is_valid(self, expression: str) -> bool:
        """Check if expression is valid."""
        return self.validate(expression)["valid"]

    def next_runs(self, expression: str, count: int = 5,
                 from_date: datetime = None) -> List[str]:
        """
        Get next run times for cron expression.

        Args:
            expression: Cron expression
            count: Number of runs to return
            from_date: Start date (defaults to now)

        Returns:
            List of datetime strings
        """
        if not CRONITER_AVAILABLE:
            return ["croniter package required for next_runs()"]

        if from_date is None:
            from_date = datetime.now()

        try:
            cron = croniter(expression, from_date)
            runs = []
            for _ in range(count):
                next_run = cron.get_next(datetime)
                runs.append(next_run.strftime("%Y-%m-%d %H:%M:%S"))
            return runs
        except Exception as e:
            return [f"Error: {str(e)}"]

    def matches(self, expression: str, dt: datetime) -> bool:
        """
        Check if datetime matches cron expression.

        Args:
            expression: Cron expression
            dt: Datetime to check

        Returns:
            True if matches
        """
        if not CRONITER_AVAILABLE:
            return False

        try:
            cron = croniter(expression, dt - timedelta(minutes=1))
            next_run = cron.get_next(datetime)
            return (next_run.year == dt.year and
                   next_run.month == dt.month and
                   next_run.day == dt.day and
                   next_run.hour == dt.hour and
                   next_run.minute == dt.minute)
        except Exception:
            return False

    def get_preset(self, name: str) -> str:
        """Get a preset cron expression."""
        if name not in self.PRESETS:
            raise ValueError(f"Unknown preset: {name}. Available: {list(self.PRESETS.keys())}")
        return self.PRESETS[name]

    def list_presets(self) -> Dict[str, str]:
        """List all available presets."""
        return self.PRESETS.copy()


def main():
    parser = argparse.ArgumentParser(
        description="Cron Expression Builder - Build and validate cron expressions"
    )

    parser.add_argument("--from-text", "-t", help="Convert natural language to cron")
    parser.add_argument("--describe", "-d", help="Describe a cron expression")
    parser.add_argument("--validate", "-v", help="Validate a cron expression")
    parser.add_argument("--next", "-n", help="Get next run times for expression")
    parser.add_argument("--count", "-c", type=int, default=5, help="Number of next runs")
    parser.add_argument("--preset", "-p", help="Use a preset schedule")
    parser.add_argument("--presets", action="store_true", help="List all presets")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive builder")

    args = parser.parse_args()

    builder = CronBuilder()

    if args.presets:
        print("\n=== Available Presets ===")
        for name, cron in builder.list_presets().items():
            desc = builder.describe(cron)
            print(f"  {name}: {cron}")
            print(f"    {desc}")

    elif args.preset:
        cron = builder.get_preset(args.preset)
        print(f"\nPreset: {args.preset}")
        print(f"Cron: {cron}")
        print(f"Description: {builder.describe(cron)}")
        if CRONITER_AVAILABLE:
            print(f"\nNext 5 runs:")
            for run in builder.next_runs(cron, count=5):
                print(f"  {run}")

    elif args.from_text:
        cron = builder.from_text(args.from_text)
        print(f"\nInput: \"{args.from_text}\"")
        print(f"Cron: {cron}")
        print(f"Description: {builder.describe(cron)}")
        if CRONITER_AVAILABLE:
            print(f"\nNext 5 runs:")
            for run in builder.next_runs(cron, count=5):
                print(f"  {run}")

    elif args.describe:
        print(f"\nCron: {args.describe}")
        print(f"Description: {builder.describe(args.describe)}")

    elif args.validate:
        result = builder.validate(args.validate)
        print(f"\nExpression: {args.validate}")
        if result["valid"]:
            print("Status: Valid")
            print(f"Description: {result['description']}")
            print(f"Fields: {result['fields']}")
        else:
            print(f"Status: Invalid")
            print(f"Error: {result['error']}")

    elif args.next:
        validation = builder.validate(args.next)
        if not validation["valid"]:
            print(f"Error: {validation['error']}")
            return

        print(f"\nCron: {args.next}")
        print(f"Description: {builder.describe(args.next)}")
        print(f"\nNext {args.count} runs:")
        for run in builder.next_runs(args.next, count=args.count):
            print(f"  {run}")

    elif args.interactive:
        print("\n=== Interactive Cron Builder ===")
        print("Enter a natural language description of your schedule:")
        text = input("> ")

        cron = builder.from_text(text)
        print(f"\nGenerated cron: {cron}")
        print(f"Description: {builder.describe(cron)}")

        if CRONITER_AVAILABLE:
            print(f"\nNext 5 runs:")
            for run in builder.next_runs(cron, count=5):
                print(f"  {run}")

        print("\nIs this correct? (y/n)")
        if input("> ").lower() != 'y':
            print("\nManually enter cron fields:")
            minute = input("Minute (0-59, */n, *): ") or "*"
            hour = input("Hour (0-23, */n, *): ") or "*"
            day = input("Day of month (1-31, *): ") or "*"
            month = input("Month (1-12, *): ") or "*"
            weekday = input("Day of week (0-6, 0=Sun, *): ") or "*"

            cron = builder.build(minute, hour, day, month, weekday)
            print(f"\nFinal cron: {cron}")
            print(f"Description: {builder.describe(cron)}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
