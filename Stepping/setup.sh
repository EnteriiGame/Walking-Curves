#!/bin/bash
# =============================================================================
# BIPED ROBOT — ONE SHOT SETUP  (Debian)
# bash setup.sh
# Safe to re-run. All output logged to setup_log.txt
# =============================================================================

set -e
cd "$(dirname "$0")"

LOG="$(pwd)/setup_log.txt"
> "$LOG"  # clear log

# Tee everything to log AND terminal
exec > >(tee -a "$LOG") 2>&1

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}  ✓ $1${NC}"; }
step() { echo -e "\n${YELLOW}════ $1 ════${NC}"; }
warn() { echo -e "${RED}  ⚠ $1${NC}"; }
info() { echo -e "  → $1"; }

echo -e "\n${YELLOW}╔══════════════════════════════════════╗"
echo -e "║     BIPED ROBOT ENVIRONMENT SETUP    ║"
echo -e "╚══════════════════════════════════════╝${NC}"
echo "  Log: $LOG"

# =============================================================================
step "1. BASE PACKAGES"
# =============================================================================
sudo apt-get update -qq
sudo apt-get install -y \
    build-essential cmake git wget curl swig \
    python3 python3-pip python3-venv python3-dev \
    libeigen3-dev libace-dev \
    liburdfdom-dev liborocos-kdl-dev
ok "Base packages ready"

# =============================================================================
step "2. YARP C++ LIBRARY"
# =============================================================================
if command -v yarp &>/dev/null; then
    ok "YARP already installed: $(yarp version 2>&1 | head -1)"
else
    info "Trying apt..."
    wget -qO /tmp/robotology_install.sh \
        https://raw.githubusercontent.com/robotology/robotology-superbuild/master/scripts/install_robotology_packages.sh \
        2>/dev/null && bash /tmp/robotology_install.sh 2>/dev/null || true
    sudo apt-get update -qq

    if sudo apt-get install -y yarp libyarp-dev 2>/dev/null; then
        ok "YARP installed via apt"
    else
        info "Building YARP from source (10-15 min)..."
        _build_yarp_from_source
    fi
fi

# =============================================================================
step "3. YARP PYTHON BINDINGS"
# =============================================================================

_build_yarp_from_source() {
    if [ ! -d "yarp_src" ]; then
        git clone --depth 1 https://github.com/robotology/yarp yarp_src
    else
        ok "yarp_src already cloned"
    fi

    info "Patching broken CMakeLists entries..."
    sed -i 's|add_subdirectory(throttleDown)|#add_subdirectory(throttleDown)|' \
        yarp_src/src/portmonitors/CMakeLists.txt 2>/dev/null || true
    sed -i 's|add_subdirectory(audioRecorder_nws_yarp)|#add_subdirectory(audioRecorder_nws_yarp)|' \
        yarp_src/src/devices/networkWrappers/CMakeLists.txt 2>/dev/null || true
    sed -i 's|add_subdirectory(RGBDSensor_nwc_yarp)|#add_subdirectory(RGBDSensor_nwc_yarp)|' \
        yarp_src/src/devices/networkWrappers/CMakeLists.txt 2>/dev/null || true

    rm -rf yarp_src/build && mkdir yarp_src/build
    cd yarp_src/build

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DYARP_COMPILE_BINDINGS=ON \
        -DCREATE_PYTHON=ON \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DYARP_COMPILE_CARRIER_PLUGINS=OFF \
        -DYARP_COMPILE_DEVICE_PLUGINS=OFF \
        -DYARP_COMPILE_PORTMONITOR_PLUGINS=OFF \
        -DYARP_COMPILE_TESTS=OFF \
        -Wno-dev

    make -j$(nproc)
    sudo make install
    sudo ldconfig
    cd ../..
    ok "YARP built and installed"
}

YARP_SO=$(find /usr /usr/local -name "_yarp*.so" 2>/dev/null | head -1)

if [ -n "$YARP_SO" ]; then
    ok "Python bindings already exist: $YARP_SO"
else
    info "Python bindings missing — building from source"

    if sudo apt-get install -y python3-yarp 2>/dev/null; then
        ok "Python bindings via apt"
        YARP_SO=$(find /usr -name "_yarp*.so" 2>/dev/null | head -1)
    else
        # Need source build — clone if not already there
        if [ ! -d "yarp_src" ]; then
            git clone --depth 1 https://github.com/robotology/yarp yarp_src
        fi

        info "Patching broken CMakeLists entries..."
        sed -i 's|add_subdirectory(throttleDown)|#add_subdirectory(throttleDown)|' \
            yarp_src/src/portmonitors/CMakeLists.txt 2>/dev/null || true
        sed -i 's|add_subdirectory(audioRecorder_nws_yarp)|#add_subdirectory(audioRecorder_nws_yarp)|' \
            yarp_src/src/devices/networkWrappers/CMakeLists.txt 2>/dev/null || true
        sed -i 's|add_subdirectory(RGBDSensor_nwc_yarp)|#add_subdirectory(RGBDSensor_nwc_yarp)|' \
            yarp_src/src/devices/networkWrappers/CMakeLists.txt 2>/dev/null || true

        rm -rf yarp_src/build && mkdir yarp_src/build
        cd yarp_src/build

        cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DYARP_COMPILE_BINDINGS=ON \
            -DCREATE_PYTHON=ON \
            -DCMAKE_INSTALL_PREFIX=/usr/local \
            -DYARP_COMPILE_CARRIER_PLUGINS=OFF \
            -DYARP_COMPILE_DEVICE_PLUGINS=OFF \
            -DYARP_COMPILE_PORTMONITOR_PLUGINS=OFF \
            -DYARP_COMPILE_TESTS=OFF \
            -Wno-dev

        make -j$(nproc)
        sudo make install
        sudo ldconfig
        cd ../..

        YARP_SO=$(find /usr/local -name "_yarp*.so" 2>/dev/null | head -1)
    fi
fi

if [ -z "$YARP_SO" ]; then
    warn "Python bindings not found after build — check setup_log.txt"
    YARP_BINDINGS_OK=false
else
    BINDING_DIR=$(dirname "$YARP_SO")
    ok "Bindings at: $BINDING_DIR"
    YARP_BINDINGS_OK=true
fi

# =============================================================================
step "4. PYTHON VENV"
# =============================================================================
if [ ! -d "venv" ]; then
    python3 -m venv venv
    ok "venv created"
else
    ok "venv already exists"
fi

source venv/bin/activate

if [ "$YARP_BINDINGS_OK" = true ]; then
    SITE=$(python -c "import site; print(site.getsitepackages()[0])")
    echo "$BINDING_DIR" > "$SITE/yarp_system.pth"
    ok "YARP bindings linked into venv"
fi

pip install --upgrade pip -q
pip install numpy -q
pip freeze > requirements.txt
ok "Python packages done"

# =============================================================================
step "5. SESSION SCRIPT"
# =============================================================================
cat > activate_robot.sh << 'EOF'
#!/bin/bash
source "$(dirname "$0")/venv/bin/activate"
echo " venv aktywowany"
echo " wpisz do konsoli"
echo " yarpserver --write &"
echo " python3 teleop.py"
echo ""
EOF
chmod +x activate_robot.sh
ok "activate_robot.sh written"

# =============================================================================
step "6. FINAL CHECKS"
# =============================================================================
echo ""
PASS=true

python --version &>/dev/null       && ok "Python:     $(python --version)"      || { warn "Python broken";          PASS=false; }
command -v yarp &>/dev/null        && ok "YARP CLI:   $(yarp version 2>&1 | head -1)" || { warn "YARP CLI not in PATH";   PASS=false; }
python -c "import yarp"  2>/dev/null && ok "import yarp:  OK"  || { warn "import yarp:  FAILED — see setup_log.txt"; PASS=false; }
python -c "import numpy" 2>/dev/null && ok "import numpy: OK"  || { warn "import numpy: FAILED";                    PASS=false; }

echo ""
if [ "$PASS" = true ]; then
    echo -e "${GREEN}╔══════════════════════════════════════╗"
    echo -e "║         ALL CHECKS PASSED  ✓         ║"
    echo -e "╚══════════════════════════════════════╝${NC}"
else
    echo -e "${YELLOW}╔══════════════════════════════════════╗"
    echo -e "║   DONE WITH WARNINGS — see above     ║"
    echo -e "║   Full log: setup_log.txt             ║"
    echo -e "╚══════════════════════════════════════╝${NC}"
fi

echo ""
echo "  Every session:  source activate_robot.sh"
echo "  Start YARP:     yarpserver --write &"
echo ""