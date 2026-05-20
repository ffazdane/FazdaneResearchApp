# Options Liquidity Local Repository

This directory is the default local storage location for Options Liquidity
Discovery snapshots.

The SQLite database files are intentionally ignored by git so development or
test snapshots do not overwrite production data during deployment.

For production, set `OPTIONS_LIQUIDITY_DB_PATH` to a persistent path outside
the git checkout, such as a mounted volume.
