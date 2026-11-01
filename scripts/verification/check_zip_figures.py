"""Compare paper.zip contents against manuscript/ files (hashes) to confirm staleness scope."""
import hashlib
import os
import zipfile


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def main() -> None:
    with zipfile.ZipFile("paper.zip") as z:
        names = z.namelist()
        print(f"paper.zip entries ({len(names)}):")
        for n in names:
            print(f"   {n:<50} {z.getinfo(n).file_size:>10}")
        print()

        print("=== hash comparison vs manuscript/ ===")
        for n in names:
            if n.endswith("/"):
                continue
            zh = sha(z.read(n))
            local = os.path.join("manuscript", n)
            status = "REPO-MISSING"
            lh = "-"
            if os.path.exists(local):
                with open(local, "rb") as f:
                    lh = sha(f.read())
                status = "MATCH" if zh == lh else "DIFF"
            print(f"{n:<50} zip={zh} repo={lh} {status}")


if __name__ == "__main__":
    main()
