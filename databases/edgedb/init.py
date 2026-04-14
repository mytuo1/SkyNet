# databases/edgedb/init.py
# Applies EdgeDB schema migrations and seeds initial data.
# Safe to run multiple times.

import asyncio
import os
import subprocess
import sys
import time

EDGEDB_DSN = os.getenv("EDGEDB_DSN", "edgedb://edgedb/main")
EDGEDB_TLS = os.getenv("EDGEDB_CLIENT_TLS_SECURITY", "insecure")


async def wait_for_edgedb(max_retries: int = 40):
    import edgedb
    for i in range(max_retries):
        try:
            client = edgedb.create_async_client(dsn=EDGEDB_DSN, tls_security=EDGEDB_TLS)
            await client.query_single("SELECT 1")
            await client.aclose()
            print(f"EdgeDB connected after {i+1} attempts")
            return
        except Exception as e:
            print(f"EdgeDB not ready ({i+1}/{max_retries}): {str(e)[:60]}")
            await asyncio.sleep(3)
    raise RuntimeError("EdgeDB never became available")


def run_migrations():
    """Run EdgeDB migrations."""
    print("Running EdgeDB migrations...")
    try:
        result = subprocess.run(
            ["edgedb", "migrate", "--dsn", EDGEDB_DSN],
            capture_output=True, text=True, timeout=60
        )
        if result.returncode == 0:
            print("Migrations applied successfully")
            print(result.stdout)
        else:
            print(f"Migration output: {result.stdout}")
            print(f"Migration stderr: {result.stderr}")
            # Not fatal — might already be up to date
    except subprocess.TimeoutExpired:
        print("Migration timed out — may already be running")
    except FileNotFoundError:
        print("edgedb CLI not found — skipping migration (run manually)")


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
    run_migrations()
    await seed_data()
    print("EdgeDB initialization complete")


if __name__ == "__main__":
    asyncio.run(main())
