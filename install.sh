git clone https://gitcode.com/Ascend/msmodeling.git
pip install -r msmodeling/requirements.txt
cp ms_pyproject.toml msmodeling/pyproject.toml

if command -v uv &>/dev/null; then
    uv sync
else
    pip install -e .
    echo "uv not found — falling back to pip install -e ."
fi
