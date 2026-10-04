# TheRock ROCm Core SDK (stable channel), per-family tarball -- the "system ROCm" that
# pwilkin/strix-halo install.sh builds the custom ROCr/HIP and llama.cpp against.
# Relocatable dist ($ORIGIN runpaths, bundled sysdeps): on NixOS the libs stay byte-for-byte untouched
# (patchelf mangles the lld-linked ones and the loader segfaults); libstdc++/libgcc_s are symlinked into
# the runpath dirs, and executables only get PT_INTERP pointed at the nix loader.
{
  lib,
  stdenv,
  fetchurl,
  gcc-unwrapped,
  glibc,
  family ? "gfx1151",
}:

let
  hashes = {
    gfx1151 = "sha256-T+q9ny2nI1LfN/bXFKVIR9P+kTwDQfviplQsEWQCS68=";
    gfx120X-all = "sha256-65nbQ0oXOP2DsMO5MxRs23ZBjzX89GR3Q/vf73boxx8=";
  };
in
stdenv.mkDerivation (finalAttrs: {
  pname = "rocm-sdk-therock-${family}";
  version = "10.0.0";

  src = fetchurl {
    url = "https://stable.repo.amd.com/rocm/core/tarball/therock-dist-linux-${family}-${finalAttrs.version}.tar.gz";
    hash = hashes.${family};
  };
  sourceRoot = ".";

  dontConfigure = true;
  dontBuild = true;
  # No stripping, no rpath shrinking: both rewrite the ELF files.
  dontStrip = true;
  dontPatchELF = true;

  installPhase = ''
    runHook preInstall
    mkdir -p $out
    cp -a . $out/
    rm -f $out/env-vars
    # Debugger builds against five system Pythons and the ROCr test suite
    # carries its own hwloc; neither is needed here and neither resolves.
    rm -f $out/bin/rocgdb-py3.*
    rm -rf $out/lib/rocrtst

    # The two host libraries the dist does not bundle, placed where its
    # $ORIGIN runpaths already look (libraries: rocm_sysdeps/lib; the LLVM
    # tree: its own lib/).
    for d in lib/rocm_sysdeps/lib lib/llvm/lib; do
      ln -s ${gcc-unwrapped.lib}/lib/libstdc++.so.6 $out/$d/
      ln -s ${gcc-unwrapped.lib}/lib/libgcc_s.so.1 $out/$d/
    done

    # Executables: interpreter only.
    while IFS= read -r f; do
      isELF "$f" || continue
      patchelf --print-interpreter "$f" >/dev/null 2>&1 || continue
      patchelf --set-interpreter ${stdenv.cc.bintools.dynamicLinker} "$f"
    done < <(find $out -type f -perm -u+x -not -name '*.so*')

    # The tools the ROCr/CLR/llama.cpp builds and the service run must still load after the interpreter
    # patch (a mangled binary fails here, not later). Test programs/profilers/flang/hipify are not checked:
    # they don't resolve on their own runpaths (expect /opt/rocm/lib).
    for f in bin/hipcc bin/hipconfig bin/rocminfo \
             lib/llvm/bin/clang-23 lib/llvm/bin/lld lib/llvm/bin/llvm-mc \
             lib/llvm/bin/llvm-objcopy lib/llvm/bin/clang-offload-bundler \
             lib/llvm/bin/clang-offload-packager lib/llvm/bin/llvm-ar; do
      ${glibc}/lib/ld-linux-x86-64.so.2 --list "$out/$f" >/dev/null \
        || { echo "required tool does not load: $f" >&2; exit 1; }
    done
    runHook postInstall
  '';

  passthru = {
    inherit family;
  };

  meta = {
    description = "ROCm Core SDK ${finalAttrs.version} (TheRock dist, ${family})";
    homepage = "https://github.com/ROCm/TheRock";
    license = lib.licenses.unfree;
    platforms = [ "x86_64-linux" ];
    sourceProvenance = [ lib.sourceTypes.binaryNativeCode ];
  };
})
