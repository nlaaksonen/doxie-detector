class Rect():
    def __init__(self, c_x, c_y, r_w, b_h, l_w=None, t_h=None):
        # Assume 0 <= c_x, c_y, <= 223,

        if l_w is None:
            l_w = r_w
        if t_h is None:
            t_h = b_h
        self.c_x = int(c_x)
        self.c_y = int(c_y)
        self.r_w = int(r_w)
        self.b_h = int(b_h)
        self.l_w = int(l_w)
        self.t_h = int(t_h)
        self._coerce()
        
    def set_rw(self, r_w):
        self.r_w = int(r_w)
        self._coerce()
        
    def set_bh(self, b_h):
        self.b_h = int(b_h)
        self._coerce()
        
    def set_lw(self, l_w):
        self.l_w = int(l_w)
        self._coerce()
        
    def set_th(self, t_h):
        self.t_h = int(t_h)
        self._coerce()
        
    def _coerce(self):
        self.set_coords(self.c_x, self.c_y, self.r_w, self.t_h, self.l_w, self.b_h)
            
    def set_coords(self, c_x, c_y, r_w, t_h, l_w, b_h):
        self.c_x = c_x
        self.c_y = c_y
        
        self.x_left = max(self.c_x - l_w, 0)
        self.x_right = min(self.c_x + r_w, 224-1)
        self.y_top = max(self.c_y - t_h, 0)
        self.y_bot = min(self.c_y + b_h, 224-1)
        
        # final height, width:
        self.h = self.y_bot - self.y_top
        self.w = self.x_right - self.x_left
        self.r_w = self.x_right - self.c_x
        self.l_w = self.c_x - self.x_left
        self.t_h = self.c_y - self.y_top
        self.b_h = self.y_bot - self.c_y