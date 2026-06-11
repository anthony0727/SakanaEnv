from scripts.make_web_demo import copy_asset


def test_copy_asset_preserves_existing_file(tmp_path):
    src = tmp_path / "src.glb"
    dst = tmp_path / "dst.glb"
    src.write_bytes(b"procedural")
    dst.write_bytes(b"meshy")

    copied = copy_asset(src, dst, force=False)

    assert copied is False
    assert dst.read_bytes() == b"meshy"


def test_copy_asset_force_overwrites_existing_file(tmp_path):
    src = tmp_path / "src.glb"
    dst = tmp_path / "dst.glb"
    src.write_bytes(b"procedural")
    dst.write_bytes(b"meshy")

    copied = copy_asset(src, dst, force=True)

    assert copied is True
    assert dst.read_bytes() == b"procedural"
