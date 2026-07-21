# diff-gaussian-rasterization のソースに <cstdint> を追加するパッチ。
# 新しめのコンパイラで std::uintptr_t 等が未定義になるのを防ぐ（MSVC でも無害）。
# setup_windows.ps1 から `python patch_rasterizer.py <raster_root>` で呼ばれる。
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else r"_build\diff-gaussian-rasterization"
cr = os.path.join(root, "cuda_rasterizer")
targets = ["rasterizer_impl.h", "rasterizer_impl.cu", "forward.h", "forward.cu",
           "backward.h", "backward.cu", "auxiliary.h", "rasterizer.h", "config.h"]
for f in targets:
    p = os.path.join(cr, f)
    if not os.path.exists(p):
        continue
    s = open(p, encoding="utf-8").read()
    if "#include <cstdint>" in s:
        print("already patched", f)
        continue
    if "#pragma once" in s:
        s = s.replace("#pragma once", "#pragma once\n#include <cstdint>", 1)
    else:
        s = "#include <cstdint>\n" + s
    open(p, "w", encoding="utf-8").write(s)
    print("patched", f)
print("patch done")
