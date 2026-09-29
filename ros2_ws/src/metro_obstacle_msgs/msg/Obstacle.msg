# Объект внутри габарита. Единицы — метры, оси — входного облака (frame_id из header массива).
float32 distance           # вдоль оси пути от лидара
float32 lateral            # смещение от оси пути, + влево
float32 height             # верх объекта над уровнем головок рельсов
float32 length             # размер вдоль пути
float32 width              # размер поперёк пути
uint32 points              # число точек лидара в объекте
float32 confidence         # 0..1
bool confirmed             # true — подтверждён по кадрам, false — кандидат одного кадра
geometry_msgs/Point position  # центр объекта в осях входного облака
float32 z_extent           # вертикальный размер точек объекта (высота рамки); у кабеля мал при большом height
