{
  description = "vllm-radiance-nix: a Nix build of vLLM for AMD RDNA4 (gfx1201 / Radeon AI PRO R9700)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.libr4d = {
    url = "git+https://codeberg.org/StillDeadcode/libr4d?rev=b9e42ab7202f53a3bc13d415f5d41481f9ca311b";
    flake = false;
  };

  outputs =
    { self, nixpkgs, libr4d }:
    let
      systems = [ "x86_64-linux" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;

      # Consume the caller's `pkgs` so a NixOS host does not evaluate a second
      # nixpkgs. `r4dSrc` defaults to this flake's libr4d input; `radianceSrc`
      # is the flake source itself -- the kernels, configs and patch chain all
      # live in this repo.
      mkVllmStack =
        { pkgs, r4dSrc ? libr4d, radianceSrc ? self }:
        import ./nix/stack.nix { inherit pkgs r4dSrc radianceSrc; };
    in
    {
      # For a NixOS module:
      #   inputs.vllm-radiance.lib.${system}.mkVllmStack { inherit pkgs; }
      lib = forAllSystems (system: { inherit mkVllmStack; });

      # Standalone: `nix build`, `nix run .#pythonEnv`, ...
      packages = forAllSystems (
        system:
        let
          stack = mkVllmStack { pkgs = nixpkgs.legacyPackages.${system}; };
        in
        {
          default = stack.pythonEnv;
          inherit (stack)
            pythonEnv
            vllm
            aiter
            torch
            libr4d
            rocmSdk
            rocmSdkCc
            runtimeLibs
            gfxArch
            ;
        }
      );

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          stack = mkVllmStack { inherit pkgs; };
        in
        {
          default = pkgs.mkShell {
            packages = [ stack.pythonEnv ];
          };
        }
      );

      formatter = forAllSystems (system: nixpkgs.legacyPackages.${system}.nixfmt-rfc-style);
    };
}
