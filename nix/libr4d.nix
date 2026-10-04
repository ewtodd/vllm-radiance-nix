{
  stdenv,
  python,
  rocmSdk,
  rocmSdkCc,
  src,
  patches ? [ ],
  gfxArch ? "gfx1201",
}:

stdenv.mkDerivation {
  pname = "libr4d";
  version = "0.5.0";

  inherit src patches;

  nativeBuildInputs = [
    rocmSdkCc
    (python.withPackages (ps: [ ps.pybind11 ]))
  ];

  dontConfigure = true;

  buildPhase = ''
    runHook preBuild
    export ROCM_PATH=${rocmSdk}
    export HIP_DEVICE_LIB_PATH=${rocmSdk}/lib/llvm/amdgcn/bitcode
    GFX_ARCH=${gfxArch} HIPCC=hipcc JOBS=$NIX_BUILD_CORES OUT=r4d.so bash ./build.sh
    runHook postBuild
  '';

  installPhase = ''
    runHook preInstall
    mkdir -p $out/${python.sitePackages}
    cp r4d.so $out/${python.sitePackages}/
    runHook postInstall
  '';

  dontStrip = true;

  meta = {
    description = "libr4d: hand-written HIP kernels for gfx1201 (RDNA4), as the r4d python extension";
    homepage = "https://codeberg.org/StillDeadcode/libr4d";
    platforms = [ "x86_64-linux" ];
  };
}
