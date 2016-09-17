
from Tkinter import *

class RecordTrajectory:

    CANVAS_WIDTH = 400
    CANVAS_HEIGHT = 400

    def __init__(self, num_trajs=2):
        self.reset()
        self.num_trajs = num_trajs


    def reset(self):
        # we will assume we have only two trajectories
        self.trajectories = [[],[]]
        self.trajectory_id = 0


    def inc_traj(self, event):
        self.trajectory_id += 1
        if (self.trajectory_id == self.num_trajs):
            self.master.destroy()


    def store_point(self, event):
        """
        Consider green to be observed and red controlled
        """
        self.trajectories[self.trajectory_id].append( (event.x, event.y) )
        color = None
        if (self.trajectory_id == 0):
            color = "#476042" # python green
        else:
            color = "#ff1a1a" # slightly light red

        x1, y1 = event.x - 1, event.y - 1
        x2, y2 = event.x + 1, event.y + 1
        self.w.create_oval(x1, y1, x2, y2, fill=color, outline=color)


    def record_trajectory(self):
        self.reset()
        self.master = Tk()
        self.master.title("Recording trajectory")
        self.w = Canvas(self.master,
                        width = RecordTrajectory.CANVAS_WIDTH,
                        height = RecordTrajectory.CANVAS_HEIGHT)
        self.w.pack(expand = YES, fill = BOTH)
        self.w.bind("<B1-Motion>", self.store_point)
        self.w.bind("<ButtonRelease-1>", self.inc_traj)
        mainloop()
        return self.trajectories


class DisplayTrajectory():

    CANVAS_WIDTH = 400
    CANVAS_HEIGHT = 400

    def __init__(self):
        self.master = Tk()
        self.master.title("Displaying trajectory")
        self.w = Canvas(self.master,
                        width = DisplayTrajectory.CANVAS_WIDTH,
                        height = DisplayTrajectory.CANVAS_HEIGHT)
        self.w.pack(expand=YES, fill=BOTH)


    def add_traj_point(self, p, is_observed_agent):
        color = None
        if (is_observed_agent):
            color = "#476042" # python green
        else:
            color = "#ff1a1a" # slightly light red

        x1, y1 = p[0] - 1, p[1] - 1
        x2, y2 = p[0] + 1, p[1] + 1
        self.w.create_oval(x1, y1, x2, y2, fill=color, outline=color)


    def add_traj_to_display(self, traj, is_observed_agent):
        for i in range(len(traj)):
            p = traj[i][0] #print p
            p = (i, p) #)
            self.add_traj_point(p, is_observed_agent)


    def show(self):
        mainloop()
