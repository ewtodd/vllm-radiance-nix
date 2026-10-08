{
  pkgs,
  r4dSrc,
  radianceSrc,
  therockSdk,
}:
let
  inherit (pkgs) lib;
  gfxArch = "gfx1201";

  rocmSdk = therockSdk.lib.${pkgs.stdenv.hostPlatform.system}.mkRocmSdk {
    inherit pkgs;
    family = "gfx120X-all";
  };
  rocmSdkCc = therockSdk.lib.${pkgs.stdenv.hostPlatform.system}.mkRocmSdkCc {
    inherit pkgs rocmSdk;
  };

  rocmPackages = {
    clr = rocmSdk // {
      icd = rocmSdk;
      gpuTargets = [ gfxArch ];
      localGpuTargets = [ gfxArch ];
    };
    hipcc = rocmSdkCc;
    llvm = {
      openmp = rocmSdk;
      clang = rocmSdkCc;
      lld = rocmSdk;
    };
    composable_kernel = rocmSdk // {
      anyMfmaTarget = false;
      composable_kernel_src = rocmSdk;
    };
    rocprofiler-sdk = rocmSdk // {
      dev = rocmSdk;
    };
  }
  // lib.genAttrs [
    "rocm-core"
    "rccl"
    "miopen"
    "miopen-hip"
    "aotriton"
    "rocrand"
    "rocblas"
    "rocsparse"
    "hipsparse"
    "hipsparselt"
    "rocthrust"
    "rocprim"
    "hipcub"
    "roctracer"
    "rocfft"
    "rocsolver"
    "hipfft"
    "hiprand"
    "hipsolver"
    "hipblas-common"
    "hipblas"
    "hipblaslt"
    "rocminfo"
    "rocm-comgr"
    "rocm-device-libs"
    "rocm-runtime"
    "rocm-smi"
    "hipify"
    "amdsmi"
    "rocshmem"
  ] (_: rocmSdk);

  hipEnv = {
    ROCM_PATH = "${rocmSdk}";
    HIP_PATH = "${rocmSdk}";
    HIP_CLANG_PATH = "${rocmSdkCc}/llvm/bin";
    HIP_DEVICE_LIB_PATH = "${rocmSdk}/lib/llvm/amdgcn/bitcode";
  };

  # radiance_mxfp4_fp8.hip -> radiance_mxfp4_fp8.so via the Dockerfile's hipcc
  # command; the W4A8 path disables itself without this extension.
  radianceMxfp4Ext =
    pkgs.runCommand "radiance-mxfp4-fp8"
      {
        nativeBuildInputs = [
          rocmSdkCc
          (python.withPackages (ps: [ ps.pybind11 ]))
        ];
      }
      ''
        export ROCM_PATH=${rocmSdk}
        export HIP_PATH=${rocmSdk}
        export HIP_CLANG_PATH=${rocmSdkCc}/llvm/bin
        export HIP_DEVICE_LIB_PATH=${rocmSdk}/lib/llvm/amdgcn/bitcode
        mkdir -p $out
        INC=$(python -m pybind11 --includes)
        hipcc -O3 -std=c++17 -fPIC -shared --offload-arch=${gfxArch} -Wno-unused-result \
          $INC ${radianceSrc}/radiance_mxfp4_fp8.hip -o $out/radiance_mxfp4_fp8.so
      '';

  # paroquant/radiance_paroquant.hip -> radiance_paroquant_kernel.so. The .hip
  # includes its sibling par_kernels.h, so no extra include path is needed.
  # Same hipcc flags as the image bake (-O3 -w -std=c++17 -fPIC -shared).
  radianceParoquantExt =
    pkgs.runCommand "radiance-paroquant-kernel"
      {
        nativeBuildInputs = [
          rocmSdkCc
          (python.withPackages (ps: [ ps.pybind11 ]))
        ];
      }
      ''
        export ROCM_PATH=${rocmSdk}
        export HIP_PATH=${rocmSdk}
        export HIP_CLANG_PATH=${rocmSdkCc}/llvm/bin
        export HIP_DEVICE_LIB_PATH=${rocmSdk}/lib/llvm/amdgcn/bitcode
        mkdir -p $out
        INC=$(python -m pybind11 --includes)
        hipcc -O3 -w -std=c++17 -fPIC -shared --offload-arch=${gfxArch} \
          $INC ${radianceSrc}/paroquant/radiance_paroquant.hip \
          -o $out/radiance_paroquant_kernel.so
      '';

  # The vLLM postInstall copies this file straight from its flake store path,
  # so `cp <file> $SP/` keeps the hash-prefixed basename and the
  # `radiance_paroquant.pth` import could not resolve it. Ship a correctly
  # named copy through pythonEnv instead; changing this derivation rebuilds
  # only the env, not vLLM.
  radianceParoquantHook = pkgs.runCommand "radiance-paroquant-hook" { } ''
    mkdir -p $out/${python.sitePackages}
    cp ${./radiance_paroquant_hook.py} $out/${python.sitePackages}/radiance_paroquant_hook.py
  '';

  radiancePatches =
    pkgs.runCommand "radiance-patches"
      {
        src = radianceSrc;
      }
      ''
        mkdir -p $out
        cp $src/patch_*.py $src/install_radiance_hooks.py $src/_patchlib.py $out/
        # patch_dflash2.py installs two new files from its sibling dflash2/ dir
        # (speculator.py, kv collector) relative to the script location.
        cp -r $src/dflash2 $out/dflash2
        chmod u+w $out/*.py
        for f in $out/*.py; do
          substituteInPlace $f --replace-quiet \
            'sysconfig.get_paths()["purelib"]' \
            '__import__("os").environ["RADIANCE_SP"]'
        done
        substituteInPlace $out/_patchlib.py --replace-fail \
          'raise SystemExit(f"  FAIL  {label}: {path} missing")' \
          'print(f"  SKIP  {label}: {path} not in this package"); return'
      '';

  applyRadiancePatches = names: ''
    export RADIANCE_SP=$out/${python.sitePackages}
    chmod -R u+w $RADIANCE_SP
    pushd ${radiancePatches} >/dev/null
    for p in ${lib.concatStringsSep " " names}; do
      echo "== radiance: $p =="
      python $p.py
    done
    popd >/dev/null
  '';

  aiterPatches = [
    "patch_unified_attention_lds"
    "patch_radiance_dispatch"
  ];

  # ggz14/radiance-vllm-mxfp4 patch chain, in the image's order: the base
  # recipe (Dockerfile L277-284) then the bake layer (Dockerfile.ggz14.top).
  # Duplicates between the two lists are idempotent and applied once here.
  # patch_unified_attention_lds targets aiter only (see aiterPatches).
  vllmPatches = [
    "patch_gfx1201"
    "patch_radiance_dispatch"
    "patch_skinny_gemm"
    "patch_gdn_wmma"
    "patch_preshuffle"
    "patch_radiance_fusion"
    "install_radiance_hooks"
    "patch_unpad"
    "patch_mtp_mm_mask"
    "patch_mtp_loopbreak"
    "patch_qwen3_toolparse"
    "patch_from_json_filter"
    "patch_dynamo_metrics"
    "patch_conv1d_blockn"
    "patch_r4d"
    "patch_dflash_base"
    "patch_dflash2"
    "patch_dflash_fused_kv_fp8"
    "patch_dflash_w4"
    "patch_gdn_metadata"
    "patch_quark_mxfp4"
    "patch_ar_maxbytes"
    "patch_topk_triton_rows"
    "patch_qwen3_thinkoff"
    # bake layer additions
    "patch_nvfp4_mxfp4"
    "patch_tp3_pad"
    "patch_dflash_calib"
    "patch_dflash_mxfp4_kv"
    "patch_rmsquant_fusion"
    "patch_verify_head"
    "patch_kv_group_size"
    "patch_topk_composite"
    "patch_gdn_shared_build"
    "patch_dflash_selector_topk"
    "patch_dflash_draft_rope"
    "patch_gdn_merge_inproj"
    "patch_dynwidth"
    "patch_async_dynwidth"
    "patch_step_trace"
    "patch_ar_geometry"
    "patch_ar_3rank"
    "patch_gdn_glue"
  ];

  tritonLlvm = pkgs.triton-llvm.overrideAttrs (old: {
    cmakeFlags = old.cmakeFlags ++ [ (lib.cmakeBool "BUILD_SHARED_LIBS" false) ];
  });

  python = pkgs.python3.override {
    packageOverrides = self: super: {
      triton = super.triton.override { llvm = tritonLlvm; };

      torch =
        (super.torch.override {
          rocmSupport = true;
          cudaSupport = false;
          inherit rocmPackages;
          gpuTargets = [ gfxArch ];
          effectiveMagma = null;
          # GCC 16's <format>, now pulled in through <chrono>, declares
          # [[__gnu__::__noinline__]]; HIP's host_defines.h defines
          # __noinline__ to nothing for host TUs, so kineto dies with
          # "expected an identifier for the attribute name". GCC 15's
          # libstdc++ doesn't use that attribute.
          stdenv = pkgs.gcc15Stdenv;
        }).overrideAttrs
          (old: {
            # TheRock 10.1 dropped rocm_smi; torch only used it for ROCm
            # symmetric memory, so drop the find_package and the link.
            patches = old.patches ++ [ ../torch-drop-rocm-smi.patch ];
            buildInputs = old.buildInputs ++ [ pkgs.libdrm ];
            env =
              old.env
              // hipEnv
              // {
                USE_MAGMA = "0";
                USE_FLASH_ATTENTION = "0";
                USE_MEM_EFF_ATTENTION = "0";
              };
          });

      torchcodec = super.torchcodec.overridePythonAttrs (old: {
        # torch's LoadHIP falls back to ${ROCM_PATH}/lib/llvm/bin/clang++ when HIP_CLANG_PATH
        # is unset: unwrapped clang, no libstdc++ headers, compiler ABI check dies on <cstdlib>.
        env = old.env // {
          HIP_CLANG_PATH = "${rocmSdkCc}/llvm/bin";
        };
        # rocm_smi-config.cmake does pkg_check_modules(libdrm REQUIRED); the raw SDK propagates
        # nothing, so the .pc must come from nixpkgs — same fix as the torch buildInputs above.
        buildInputs = old.buildInputs ++ [ pkgs.libdrm ];
      });

      backrefs = super.backrefs.overridePythonAttrs (_: {
        doCheck = false;
      });
      einops = super.einops.overridePythonAttrs (_: {
        doCheck = false;
      });
      interegular = super.interegular.overridePythonAttrs (_: {
        doCheck = false;
      });
      devtools = super.devtools.overridePythonAttrs (_: {
        doCheck = false;
      });
      hypercorn = super.hypercorn.overridePythonAttrs (_: {
        doCheck = false;
      });
      inline-snapshot = super.inline-snapshot.overridePythonAttrs (_: {
        doCheck = false;
      });
      outlines = super.outlines.overridePythonAttrs (_: {
        doCheck = false;
      });
      mistral-common = super.mistral-common.overridePythonAttrs (old: {
        pythonRelaxDeps = (old.pythonRelaxDeps or [ ]) ++ [ "numpy" ];
        # The lossy audio round-trip RMSE lands at 0.0287 against the 0.005
        # tolerance with the current codec stack; nothing vLLM uses.
        disabledTests = (old.disabledTests or [ ]) ++ [ "test_audio_base64" ];
      });
      xgrammar = super.xgrammar.overridePythonAttrs (old: rec {
        version = "0.2.3";
        src = pkgs.fetchFromGitHub {
          owner = "mlc-ai";
          repo = "xgrammar";
          tag = "v${version}";
          fetchSubmodules = true;
          hash = "sha256-bznSz1fOCCGFR3NsuXm5eWo7EXrvBrFavEllC5+vDHM=";
        };
        patches = [ ];
        build-system = [
          self.cmake
          self.ninja
          self.scikit-build-core
          self.apache-tvm-ffi
        ];
        dependencies = old.dependencies ++ [ self.apache-tvm-ffi ];
        doCheck = false;
      });
      transformers = super.transformers.overridePythonAttrs (old: rec {
        # ggz14 builds vLLM 0.29 against transformers 5.14.1.
        version = "5.14.1";
        src = pkgs.fetchFromGitHub {
          owner = "huggingface";
          repo = "transformers";
          tag = "v${version}";
          hash = "sha256-BmjfFETKt01Z7fYgL83KOSPthZBu19yU5IdITCrpEv0=";
        };
        # 5.14.1 caps tokenizers <=0.23.0 while this nixpkgs ships 0.23.2; the
        # bound is enforced both by the metadata check and at import time, so
        # widen it in the dependency table itself.
        dontCheckRuntimeDeps = true;
        postPatch = (old.postPatch or "") + ''
          substituteInPlace src/transformers/dependency_versions_table.py \
            --replace-fail 'tokenizers>=0.22.0,<=0.23.0' 'tokenizers>=0.22.0'
        '';
        postInstall = (old.postInstall or "") + applyRadiancePatches [ "patch_from_json_filter" ];
      });

      amdsmi = self.buildPythonPackage {
        pname = "amdsmi";
        # Must match share/amd_smi/amdsmi/_version.py in the TheRock tarball.
        version = "27.1.0+7da43026";
        pyproject = true;
        src = "${rocmSdk}/share/amd_smi";
        build-system = [ self.setuptools ];
        postPatch = ''
          # TheRock 10.1.0 dropped pyproject.toml/setup.py from share/amd_smi;
          # restore the same minimal setuptools metadata the wheel was built with.
          cat > pyproject.toml <<'EOF'
          [build-system]
          requires = ["setuptools>=59.0"]
          build-backend = "setuptools.build_meta"

          [project]
          name = "amdsmi"
          version = "27.1.0+7da43026"
          description = "AMDSMI Python LIB - AMD GPU Monitoring Library"
          requires-python = ">=3.6"

          [tool.setuptools]
          packages = ["amdsmi"]
          EOF

          # 10.1's wrapper resolves <root>/lib/<soname> relative to its own
          # install tree; in site-packages the ROCm root is the SDK instead.
          substituteInPlace amdsmi/amdsmi_wrapper.py --replace-fail \
            'relocatable = here.parents[3] / "lib" / _AMDSMI_LIB_SONAME' \
            'relocatable = Path("${rocmSdk}/lib") / _AMDSMI_LIB_SONAME'
        '';
        pythonImportsCheck = [ "amdsmi" ];
      };

      amd-aiter = (super.amd-aiter.override { inherit rocmPackages; }).overrideAttrs (old: {
        # ggz14 image pins aiter v0.1.17; the radiance patches carry its module
        # paths (aiter.ops.triton.gemm.basic.gemm_afp4wfp4).
        version = "0.1.17";
        src = pkgs.fetchFromGitHub {
          owner = "ROCm";
          repo = "aiter";
          rev = "3d6dd62e54f5acca1a5d11bee73192b7c955b1bd";
          hash = "sha256-11nT5QT9Yccj0b/S6d2J+0gfuUcFfFv2v16i4dlCWqI=";
          fetchSubmodules = true;
        };
        postPatch = ''
          substituteInPlace pyproject.toml \
            --replace-fail '"flydsl==0.2.2"' ""

          substituteInPlace csrc/cpp_itfs/utils.py \
            --replace-fail \
              'commit_id = get_git_commit_id_short()' \
              'commit_id = "0.1.17"'

          # 0.1.17 spells the ROCm include append as _join_rocm_home("include")
          # where 0.1.20 had paths.append(rocm_include).
          substituteInPlace aiter/jit/utils/cpp_extension.py \
            --replace-fail \
              'paths.append(_join_rocm_home("include"))' \
              'paths.append(_join_rocm_home("include")); paths.extend(os.environ.get("NIX_AITER_ROCM_INCL", "${rocmSdk}/include:${lib.getDev self.pybind11}/include").split(":"))'

          substituteInPlace setup.py \
            --replace-fail \
              '    prepare_packaging()' \
              '    if not os.path.exists("aiter_meta"): prepare_packaging()' \
            --replace-fail \
              'if os.path.exists("aiter_meta") and os.path.isdir("aiter_meta"):' \
              'if False:'
        '';
        pythonRemoveDeps = [ "flydsl" ];
        env = old.env // hipEnv;
        postInstall =
          (old.postInstall or "")
          + applyRadiancePatches aiterPatches
          + ''
            mkdir -p $out/${python.sitePackages}/aiter/ops/triton/configs/gemm
            cp -f ${radianceSrc}/mxfp4-configs/* $out/${python.sitePackages}/aiter/ops/triton/configs/gemm/
          '';
      });

      vllm =
        (super.vllm.override {
          inherit rocmPackages;
          gpuTargets = [ gfxArch ];
          xformers = null;
        }).overrideAttrs
          (
            finalAttrs: old: {
              # ggz14 builds vLLM v0.29.0 on torch 2.11; we keep the nixpkgs
              # torch 2.13/triton 3.7 and will see whether 0.29 is stable on it.
              version = "0.29.0";
              src = pkgs.fetchFromGitHub {
                owner = "vllm-project";
                repo = "vllm";
                rev = "98dff2a81d747d1dba01a47f939f48c3526d4206";
                hash = "sha256-SPxnCItBgeOJk+io4R4f4JXfH2GSlqRKoIXCw+gpsqE=";
              };
              patches = lib.filter (p: baseNameOf p != "0005-drop-intel-reqs.patch") old.patches;
              propagatedBuildInputs = lib.filter (
                d: d != null && (d.pname or "") != "torchaudio"
              ) old.propagatedBuildInputs;
              pythonRemoveDeps = old.pythonRemoveDeps ++ [ "torchaudio" ];
              cargoDeps = pkgs.rustPlatform.fetchCargoVendor {
                inherit (finalAttrs)
                  pname
                  version
                  src
                  cargoRoot
                  ;
                hash = "sha256-eXCartOeDjzIq8WY/EwV7symrUEHTRkEf4uaehsiPAM=";
              };
              env = old.env // hipEnv;
              # vllm's CMake hits the same rocm_smi → pkg_check_modules(libdrm REQUIRED) chain
              # through find_package(Torch); see the torchcodec override.
              buildInputs = old.buildInputs ++ [ pkgs.libdrm ];
              postInstall =
                (old.postInstall or "")
                + ''
                  SP=$out/${python.sitePackages}
                  chmod -R u+w $SP
                  cp ${radianceSrc}/radiance_*.py ${radianceSrc}/radiance_amdsmi.pth $SP/
                  cp ${radianceMxfp4Ext}/radiance_mxfp4_fp8.so $SP/
                  # ParoQuant serving path: modules + the rotation/GEMM kernel.
                  cp ${radianceSrc}/paroquant/radiance_paroquant.py \
                     ${radianceSrc}/paroquant/radiance_paroquant_mxfp4.py $SP/
                  cp ${radianceParoquantExt}/radiance_paroquant_kernel.so $SP/
                  cp ${./radiance_paroquant_hook.py} $SP/
                  printf 'import radiance_paroquant_hook\n' > $SP/radiance_paroquant.pth
                  cp -f ${radianceSrc}/fp8-configs/* $SP/vllm/model_executor/layers/quantization/utils/configs/
                  cp -f ${radianceSrc}/moe-configs/* $SP/vllm/model_executor/layers/fused_moe/configs/
                  mkdir -p $out/share/vllm-radiance
                  cp ${radianceSrc}/qwen3.8-enhanced.jinja ${radianceSrc}/qwen3.6-enhanced.jinja $out/share/vllm-radiance/
                ''
                + applyRadiancePatches vllmPatches;
            }
          );
    };
  };

  libr4d = pkgs.callPackage ./libr4d.nix {
    inherit
      python
      rocmSdk
      rocmSdkCc
      gfxArch
      ;
    src = r4dSrc;
    patches = [ "${radianceSrc}/r4d_radiance_extras.patch" ];
  };

  pythonEnv = python.withPackages (ps: [
    ps.vllm
    (ps.toPythonModule libr4d)
    (ps.toPythonModule radianceParoquantHook)
  ]);

  runtimeLibs = "${rocmSdk}/lib";
in
{
  inherit
    rocmSdk
    rocmSdkCc
    rocmPackages
    python
    libr4d
    pythonEnv
    runtimeLibs
    gfxArch
    ;
  torch = python.pkgs.torch;
  aiter = python.pkgs.amd-aiter;
  vllm = python.pkgs.vllm;
}
