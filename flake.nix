{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    pyproject-nix = {
      url = "github:nix-community/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = {
    self,
    nixpkgs,
    pyproject-nix,
    ...
  }: let
      ## List of nixpkgs used in our development environment
      developmentPackages = pkgs:
        (with pkgs; [
          gitFull
          pre-commit
          lazygit
          sqlite
          ghidra
          singularity
          fish
          ncurses
          which
          stdenv
          curl
        ]);
      libraryDependencies = pkgs:
        (with pkgs; [
          stdenv.cc.cc
          gcc-unwrapped
          glibc
          glib
          openssl
          libpng
          dbus
          zstd
        ]);
      ## Lists of OS platform architectures and oses we support
      supportedSystems = [ "x86_64-linux" "x86_64-darwin" "aarch64-linux" "aarch64-darwin" ];
      ## Helper functions
      ### function to create an attrset for all supported systems given a fn of the system
      forAllSystems = nixpkgs.lib.genAttrs supportedSystems;
      ### an attrset mapping nix systems to their nixpkgs attrset
      allPkgs = forAllSystems (system: nixpkgs.legacyPackages.${system});
  in {
    # define available nixpkgs exposed by this flake
    packages = forAllSystems (system: let
      pkgs = allPkgs.${system};
      python = pkgs.python310.override {
        # overriding custom nixpkgs python modules
        self = pkgs.python310;
        packageOverrides = final: prev: {
          # pre-commit in nixpkgs is an application, not a module, but it's really both
          pre-commit = final.toPythonModule pkgs.pre-commit;
        };
      };

      # define the python project and read constraints
      undertaleProject = pyproject-nix.lib.project.loadRequirementsTxt {
        projectRoot = ./.;
        requirements = pkgs.lib.readFile ./constraints.txt;
      };

      undertalePkgDeps = undertaleProject.renderers.withPackages { inherit python; };

      undertaleConstraintErrors = undertaleProject.validators.validateVersionConstraints {
        inherit python;
      };

      validatedConstraints = (pkgs.lib.debug.traceValSeq undertaleConstraintErrors) == {};

      undertalePkgEnv = python.withPackages (undertalePkgDeps);
    in {
      default = self.packages.${system}.undertale-venv;

      # NOTE: Currently not validating constraint matching because nix python packages lag slightly on several packages, but not apparently in a critical way.  will address as needed.  use package undertale-strict if you want to enforce version freezes

      # export python environments containing all undertale python dependancies
      undertale-environment =  undertalePkgEnv;
      undertale-environment-strict = assert validatedConstraints; undertalePkgEnv;

      #export a nixpkg for the undertale python package itself
      # NOTE: should we compute just the non-development requirements in the environment
      undertale = python.pkgs.buildPythonPackage {
        name = "undertale";
        src = ./.;
        propagatedBuildInputs = [ undertalePkgEnv ];
      };

      undertale-strict = assert validatedConstraints; self.packages.${system}.undertale;

      ## Defines a buildable singularity image containing devtools and dependencies
      singularity-image = pkgs.singularity-tools.buildImage {
        name = "undertale";
        memSize = 2048;
        diskSize = 15000;
        contents = [ undertalePkgEnv ]
          ++ (with pkgs; [ 
            bat
            helix
            stdenv
          ]) 
          ++ (developmentPackages pkgs);
      };

      ## Defines a docker image containing this build of undertale and it's dependencies
      docker-image = pkgs.dockerTools.buildLayeredImage (let
        pkgsLinux = allPkgs."x86_64-linux";
        pkgsSelf = self.packages."x86_64-linux";
      in  {
        name = "undertale";
        tag = "latest";
        contents = [ pkgsSelf.undertale ] ++ (developmentPackages pkgsLinux);
        config = {
          Cmd = "${pkgsSelf.undertale-environment}/bin/python3";
        };
      });
    });

    # define available developer shells exposed by this flake
    devShells = forAllSystems (system:
    let
      pkgs = allPkgs.${system};
      python = pkgs.python310;
      wrappedPython = (pkgs.writeShellScriptBin "python" ''
        export LD_LIBRARY_PATH=$NIX_LD_LIBRARY_PATH
        SCRIPT_DIR=$(${pkgs.coreutils}/bin/dirname $(${pkgs.coreutils}/bin/realpath -s "$0"))
        exec -a "$SCRIPT_DIR/python" "${python}/bin/python" "$@"
      '');
      venvDir = "./.venv";
    in rec {
        default = undertale-venv;
        undertale-pkg = pkgs.mkShell {
          packages = [ 
            self.packages.${system}.undertale-environment
          ] ++ (developmentPackages pkgs);
        };
        undertale-venv = pkgs.mkShell {
          NIX_LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (libraryDependencies pkgs);
          packages = (developmentPackages pkgs) ++ (libraryDependencies pkgs);
          buildInputs = [
            wrappedPython
          ] ++ (libraryDependencies pkgs);
          shellHook = ''
            SOURCE_DATE_EPOCH=$(date +%s)

            if [ -d ${venvDir} ]; then
              echo "${venvDir} already exists, skipping venv creation"
            else
              echo "Creating new venv environment in ${venvDir}"
              ${wrappedPython}/bin/python -m venv "${venvDir}"
            fi

            source "${venvDir}/bin/activate"

            #fancy pipe tricks because pip is too verbose
            set -o pipefail; pip install -r constraints.txt | { grep -v "already satisfied" || :; }
          '';
        };
      }
    );
  };
}
