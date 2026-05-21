import logging
from pathlib import Path

LOG_PATH = Path(__file__).parent.parent / "app.log"
logging.basicConfig(
    filename=str(LOG_PATH),
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    encoding="utf-8",
)
log = logging.getLogger("AutoMLEngine")
log.info("=== Analytics Auto-ML Universal Hybrid Engine V17.0 เริ่มทำงาน ===")