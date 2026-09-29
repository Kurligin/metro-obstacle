# Итог детектора по одному обработанному кадру лидара.
# header (время и frame_id) копируется из входного PointCloud2.
std_msgs/Header header

# Значения level: 0 — габарит свободен; 1 — кандидат (что-то видно, по кадрам ещё
# не подтверждено); 2 — препятствие подтверждено по нескольким кадрам.
uint8 LEVEL_CLEAR=0
uint8 LEVEL_CANDIDATE=1
uint8 LEVEL_CONFIRMED=2

bool obstacle              # true, если в габарите подтверждённое препятствие (level == 2)
float32 distance           # м вдоль пути до ближайшего подтверждённого; NaN, если нет
float32 confidence         # 0..1: level 2 — у ближайшего подтверждённого; level 1 — максимум по кандидатам; 0, когда свободно
uint8 level                # LEVEL_* выше
float32 visible_range      # м вдоль пути — докуда коридор габарита реально виден
float32 processing_ms      # время обработки кадра в ноде, мс (разбор облака + детектор)
string mode                # режим детектора: default | strict | soft
bool calibrated            # false — ядро ещё калибруется (ось, полотно); тревоги не подтверждаются
