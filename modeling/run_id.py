"""
실험 폴더명 생성 유틸리티

날짜_사용자명_번호 형식으로 실험 폴더명을 자동 생성.
이름은 outputs/.username에 저장되며 이후 생략 가능.

생성 예시: 0403_leehyn_1, 0403_leehyn_2

# 최초 1회 이름 등록 (이후 --user 생략 가능)
python3 modeling/train.py --user leehyn --data ...

# 이후 실행 (저장된 이름 자동 사용)
python3 modeling/train.py --data ...
python3 modeling/feature_selection_experiment.py

"""

import os
import re
from datetime import datetime

_ROOT            = os.path.join(os.path.dirname(__file__), "..")
DIR_OUTPUTS      = os.path.join(_ROOT, "outputs")
_USERNAME_FILE   = os.path.join(DIR_OUTPUTS, ".username")


def resolve_username(user_arg: str | None) -> str:
    """
    사용자명을 결정하고 파일에 저장.

    user_arg가 주어지면 해당 값을 파일에 저장 후 반환.
    주어지지 않으면 저장된 값을 불러옴. 둘 다 없으면 오류.
    """
    if user_arg:
        os.makedirs(DIR_OUTPUTS, exist_ok=True)
        with open(_USERNAME_FILE, "w", encoding="utf-8") as f:
            f.write(user_arg.strip())
        return user_arg.strip()

    if os.path.isfile(_USERNAME_FILE):
        with open(_USERNAME_FILE, encoding="utf-8") as f:
            name = f.read().strip()
        if name:
            return name

    raise ValueError(
        "--user 인수로 사용자명을 지정하세요 (이후 자동 저장됨)\n"
        "  예) python modeling/train.py --user leehyn --data ..."
    )


def next_run_id(base_dir: str, prefix: str) -> str:
    """
    base_dir 안에서 prefix로 시작하는 폴더 중 가장 큰 번호를 찾아
    prefix_{N+1} 반환.

    prefix 예시: "0403_leehyn"
    """
    if not os.path.exists(base_dir):
        return f"{prefix}_1"

    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    nums = []
    for item in os.listdir(base_dir):
        m = pattern.match(item)
        if m and os.path.isdir(os.path.join(base_dir, item)):
            nums.append(int(m.group(1)))

    next_num = max(nums) + 1 if nums else 1
    return f"{prefix}_{next_num}"


def make_run_id(base_dir: str, user_arg: str | None) -> str:
    """
    날짜_사용자명_번호 형식의 실험 ID 반환.

    base_dir  : 실험 폴더가 위치할 상위 디렉터리 (예: outputs/, outputs/feature_selection/)
    user_arg  : CLI에서 받은 --user 값 (없으면 저장된 값 사용)
    """
    username = resolve_username(user_arg)
    date_str = datetime.now().strftime("%m%d")
    prefix   = f"{date_str}_{username}"
    return next_run_id(base_dir, prefix)


def write_summary(exp_dir: str, markdown: str) -> None:
    """실험 폴더에 summary.md 저장"""
    path = os.path.join(exp_dir, "summary.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(markdown)
    print(f"요약 저장: {path}")
