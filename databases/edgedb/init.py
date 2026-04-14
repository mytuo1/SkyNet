# databases/edgedb/init.py
# Applies EdgeDB schema migrations and seeds initial data.
# Safe to run multiple times.

import asyncio
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, urlunparse

EDGEDB_DSN = os.getenv("EDGEDB_DSN", "edgedb://edgedb/main")
EDGEDB_TLS = os.getenv("EDGEDB_CLIENT_TLS_SECURITY", "insecure")

# Schema file location — /init/edgedb/schema/schema.esdl inside the container
# (databases/ is mounted at /init/)
SCHEMA_DIR = Path(__file__).resolve().parent / "schema"


def _base_dsn(dsn: str) -> str:
    """Strip the database name from a DSN to connect to the default database."""
    parsed = urlparse(dsn)
    return urlunparse(parsed._replace(path=""))


def _db_name(dsn: str) -> str:
    """Extract the database name from a DSN (path component without leading /)."""
    name = urlparse(dsn).path.lstrip("/") or "main"
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"Invalid database name in DSN: {name!r}")
    return name


async def wait_for_edgedb(max_retries: int = 40):
    import edgedb

    base = _base_dsn(EDGEDB_DSN)
    db = _db_name(EDGEDB_DSN)

    # 1) Wait for the EdgeDB server to be reachable (default database).
    for i in range(max_retries):
        try:
            client = edgedb.create_async_client(dsn=base, tls_security=EDGEDB_TLS)
            await client.query_single("SELECT 1")
            await client.aclose()
            print(f"EdgeDB reachable after {i+1} attempts")
            break
        except Exception as e:
            print(f"EdgeDB not ready ({i+1}/{max_retries}): {str(e)[:60]}")
            await asyncio.sleep(3)
    else:
        raise RuntimeError("EdgeDB never became available")

    # 2) Ensure the target database exists.
    try:
        client = edgedb.create_async_client(dsn=EDGEDB_DSN, tls_security=EDGEDB_TLS)
        await client.query_single("SELECT 1")
        await client.aclose()
        print(f"Database '{db}' exists")
        return
    except Exception:
        pass

    print(f"Database '{db}' not found — creating...")
    try:
        client = edgedb.create_async_client(dsn=base, tls_security=EDGEDB_TLS)
        await client.execute(f"CREATE DATABASE {db}")
        await client.aclose()
        print(f"Created database '{db}'")
    except Exception as e:
        print(f"Database creation note: {e}")

    # 3) Verify the target database is usable.
    for i in range(10):
        try:
            client = edgedb.create_async_client(dsn=EDGEDB_DSN, tls_security=EDGEDB_TLS)
            await client.query_single("SELECT 1")
            await client.aclose()
            print(f"Database '{db}' ready")
            return
        except Exception as e:
            print(f"Waiting for '{db}' ({i+1}/10): {str(e)[:60]}")
            await asyncio.sleep(2)

    raise RuntimeError(f"Database '{db}' never became available")


def run_migrations():
    """Run EdgeDB migrations via CLI if available."""
    print("Running EdgeDB migrations...")
    try:
        result = subprocess.run(
            ["edgedb", "migrate", "--dsn", EDGEDB_DSN,
             "--tls-security", EDGEDB_TLS],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            print("Migrations applied successfully")
            print(result.stdout)
            return True
        else:
            print(f"Migration output: {result.stdout}")
            print(f"Migration stderr: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("Migration timed out — may already be running")
        return False
    except FileNotFoundError:
        print("edgedb CLI not found — will apply schema via Python SDK")
        return False


async def apply_schema_via_sdk():
    """Apply schema.esdl programmatically when the edgedb CLI is not available."""
    import edgedb

    schema_file = SCHEMA_DIR / "schema.esdl"
    if not schema_file.exists():
        print(f"Schema file not found at {schema_file} — skipping schema apply")
        return

    schema_sdl = schema_file.read_text()
    print(f"Applying schema from {schema_file} via Python SDK...")

    client = edgedb.create_async_client(dsn=EDGEDB_DSN, tls_security=EDGEDB_TLS)
    try:
        # Check if schema is already applied by testing for a known type
        try:
            await client.query_single("SELECT count(default::Episode)")
            print("Schema already applied — skipping")
            return
        except edgedb.errors.InvalidReferenceError:
            pass  # Type doesn't exist yet — apply schema

        await client.execute(f"""
            START MIGRATION TO {{
                {schema_sdl}
            }};
            POPULATE MIGRATION;
            COMMIT MIGRATION;
        """)
        print("Schema applied successfully via Python SDK")
    except Exception as e:
        print(f"Schema apply error: {e}")
    finally:
        await client.aclose()


async def seed_data():
    """Seed initial data if database is empty."""
    import edgedb
    client = edgedb.create_async_client(dsn=EDGEDB_DSN, tls_security=EDGEDB_TLS)

    try:
        count = await client.query_single("SELECT count(Episode)")
        print(f"Episodes in DB: {count}")

        if count == 0:
            print("Database is empty — inserting seed concepts...")
            seed_concepts = [
                "learning", "reasoning", "memory", "perception",
                "language", "mathematics", "logic", "creativity"
            ]
            for concept in seed_concepts:
                try:
                    await client.execute("""
                        INSERT Concept { name := <str>$name }
                        UNLESS CONFLICT ON .name
                    """, name=concept)
                except Exception:
                    pass
            print(f"Seeded {len(seed_concepts)} base concepts")
        else:
            print("Database already has data — skipping seed")

    except Exception as e:
        print(f"Seed error: {e}")
    finally:
        await client.aclose()


async def main():
    await wait_for_edgedb()
    cli_ok = run_migrations()
    if not cli_ok:
        await apply_schema_via_sdk()
    await seed_data()
    print("EdgeDB initialization complete")


if __name__ == "__main__":
    asyncio.run(main())
