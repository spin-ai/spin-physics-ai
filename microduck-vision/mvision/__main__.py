"""`python -m mvision <cmd>` —— 等价于 `python -m mvision.cli <cmd>`。"""

from .cli import main

raise SystemExit(main())
