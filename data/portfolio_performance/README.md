# Portfolio Performance Database

This directory is the default local development storage location for Portfolio
Performance snapshots.

SQLite database files are intentionally ignored by git so development or test
snapshots do not overwrite production data during deployment.

For production, set `PORTFOLIO_PERFORMANCE_DB_PATH` to a persistent path
outside the git checkout, such as a mounted volume:

```text
PORTFOLIO_PERFORMANCE_DB_PATH=/mnt/data/portfolio_performance.sqlite
```

If production uses the default `data/portfolio_performance/portfolio_performance.sqlite`
inside the app repository, platforms with ephemeral app storage may wipe the
database on reboot or redeploy.
