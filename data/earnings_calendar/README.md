# Earnings Calendar Database

This folder stores the local SQLite cache for the Earnings Calendar module.

The app creates `earnings_calendar.sqlite` here on first real earnings fetch.
SQLite files are intentionally ignored by git because they are local runtime data.

For production, set `EARNINGS_CALENDAR_DB_PATH` to a persistent path outside
the git checkout, such as a mounted volume:

```text
EARNINGS_CALENDAR_DB_PATH=/mnt/data/earnings_calendar.sqlite
```

If production uses the default `data/earnings_calendar/earnings_calendar.sqlite`
inside the app repository, platforms with ephemeral app storage may wipe the
database on reboot or redeploy.
