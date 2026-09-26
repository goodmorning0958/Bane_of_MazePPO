class LiDAR:
    def CheckIntersection(self, Wall, robot_pos, dx, dy):
        # get the parametric vector equations
        x1, y1 = Wall[0][0], Wall[0][1]
        x2, y2 = Wall[1][0], Wall[1][1]
        x3, y3 = robot_pos[0], robot_pos[1]

        den = dx * (y2 - y1) - dy * (x2 - x1)

        if den == 0:
           return float('inf')

        # check if they intersect
        u = (dx * (y3 - y1) - dy * (x3 - x1)) / den
        t = ((x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)) / den

        eps = 1e-5

        if (-eps <= u <= 1 + eps) and (t >= -eps):
            return t
        else:
            return float('inf')

    def CastRay(self, robot_pos, walls, dx, dy):
        if walls is None or len(walls) == 0:
            return float('inf')

        best_d = float('inf')
        for wall in walls:
            distance = self.CheckIntersection(wall, robot_pos, dx, dy)
            if distance < best_d:
               best_d = distance
        return best_d