'''
This project is a GUI for the clay_dep_cont model

It is only the GUI and cannot run the model by itself

@author: shahar

'''

from tkinter import *
from tkinter import filedialog as fd
import pickle
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import sys
model_dir = "/home/sb/Desktop/to_exe/clay_dep_cont/"
if model_dir not in sys.path:
	sys.path.append(model_dir)

from simulation import Simulation, particle_colors, markers

root = Tk()
root.geometry("150x210+1000+100")
root.title("Main Menu")

class CreateToolTip(object):

	def __init__(self, widget, text='widget info'):
		self.widget = widget
		self.text = text
		self.widget.bind("<Enter>", self.enter)
		self.widget.bind("<Leave>", self.close)

	def enter(self, event=None):
		x = y = 0
		x, y, cx, cy = self.widget.bbox("insert")
		x += self.widget.winfo_rootx() + 25
		y += self.widget.winfo_rooty() + 20
		# creates a toplevel window
		self.tw = Toplevel(self.widget)
		# Leaves only the label and removes the app window
		self.tw.wm_overrideredirect(True)
		self.tw.wm_geometry("+%d+%d" % (x, y))
		label = Label(self.tw, text=self.text, justify='left',
					   background='yellow', relief='solid', borderwidth=1,
					   font=("times", "8", "normal"))
		label.pack(ipadx=1)

	def close(self, event=None):
		if self.tw:
			self.tw.destroy()


#VARIABLES

##############HYDRO
bedform_celerity_v = DoubleVar()
bedform_celerity_v.set(60)

flow_vel_v = DoubleVar()
flow_vel_v.set(0.31)

depth_of_sediment_v = DoubleVar()
depth_of_sediment_v.set(20)

specific_storage_v = DoubleVar()
specific_storage_v.set(0.33)

water_depth_v = DoubleVar()
water_depth_v.set(12)

porosity_v = DoubleVar()
porosity_v.set(0.33)

global_k_v = DoubleVar()
global_k_v.set(0.12)

global_k_x_v = DoubleVar()
global_k_x_v.set('x')

global_k_y_v = DoubleVar()
global_k_y_v.set('y')

radio_for_global = IntVar()

is_clogging = BooleanVar()
is_clogging.set(False)

clogging_type = StringVar()
clogging_type.set('None')

tri_wavelength = DoubleVar()
tri_wavelength.set(15)

tri_height = DoubleVar()
tri_height.set(1.5)

tri_crest = DoubleVar()
tri_crest.set(10)

radio_for_tri = IntVar()
##############HYDRO
##############Model and sim
domain_length = DoubleVar()
domain_length.set(150)

domain_nodesx = IntVar()
domain_nodesx.set(150)

domain_height = DoubleVar()
domain_height.set(25)

domain_nodesz = IntVar()
domain_nodesz.set(150)


domain_width = DoubleVar()
domain_width.set(30)

domain_dt = DoubleVar()
domain_dt.set(60)

domain_dx = DoubleVar()
domain_dx.set(domain_length.get()/domain_nodesx.get())

domain_dz = DoubleVar()
domain_dz.set(domain_height.get()/domain_nodesz.get())

sim_duration = DoubleVar()
sim_duration.set(2500)

bedform_dx = DoubleVar()
bedform_dx.set(0.01)

frame_duration = DoubleVar()
frame_duration.set(100)

head_margin = DoubleVar()
head_margin.set(5)

include_surface_water_conc = BooleanVar()
include_surface_water_conc.set(False)

dynamic_flow = BooleanVar()
dynamic_flow.set(False)
#############Model and sim


############BOUND COND

radio_for_flux = IntVar()

flux = DoubleVar()
flux.set(20)

flux_type = StringVar()
flux_type.set("cm/day")

is_dynamic_flow_velocity = BooleanVar()
is_dynamic_flow_velocity.set(False)

############BOUND COND


#############particles
particles = []

particle_name_v = StringVar()
particle_name_v.set("Particle 1")
settling_vel_v = DoubleVar()
filtration_c_v = DoubleVar()
diffusion_c_v = DoubleVar()
color_in_plots = StringVar()
particle_value = DoubleVar()
particle_unit = StringVar()
leave_trail = BooleanVar()

particle_menu_ = StringVar()

to_del = IntVar()
to_del.set(1)
to_edit = IntVar()
to_edit.set(1)
#############particles

##############PROJECT
project_name = StringVar()
project_name.set("New Project")
project_type = StringVar()
project_type.set(0)
##################PROJECT
def ps():
	pass

def open_text_file():
	global particles
	# file type
	filetypes = (
		('Projects', '*.pkl'),
		('All files (placeholder)', '*.*')
	)

	name = fd.askopenfilename(filetypes=filetypes)
	#domain_width.set(int(f.read()))

	with open(name, 'rb') as p:
		config = pickle.load(p)
		print(config)

	bedform_celerity_v.set(config["celerity_cm_min"])
	flow_vel_v.set(config["flow_vel"])
	depth_of_sediment_v.set(config["depth_of_sediment"])
	specific_storage_v.set(config["specific_storage"])
	water_depth_v.set(config["water_depth_cm"])
	porosity_v.set(config["porosity"])

	#Conductivity
	global_k_v.set(config["Ks_sand"])
	is_clogging.set(config['feedback'])
	clogging_type.set(config["clogging_type"])

	#Bedform Shape
	tri_wavelength.set(config["tri_wavelength"])
	tri_height.set(config['tri_height'])
	tri_crest.set(config['tri_crest'])

	config["x_min"] = 0
	domain_length.set(config["x_max"])
	domain_nodesx.set(config["J"]) # note: as the model is written, J actually refers to the number of segments in the x-direction, so the number of x nodes is J+1
	domain_dx.set(config["dx"])

	config["z_min"] = 0
	domain_height.set(config["z_max"])
	domain_nodesz.set(config["I"]) # same comment as for J above
	domain_dz.set(config["dz"])

	domain_width.set(config["domain_width"])
	bedform_dx.set(config["bedform_dx"])

	#Time
	sim_duration.set(config["sim_duration_min"])
	domain_dt.set(config['dt'])
	domain_dx.set(config["displacement_per_timestep_cm"])

	#Animation
	frame_duration.set(config["frame_duration"])
	head_margin.set(config["head_margin_cm"]) # since head behaves anomalously near side boundary, leave a margin at either side boundary when doing particle tracking calculations

	#Extras
	include_surface_water_conc.set(config["include_surface_water_conc"])
	dynamic_flow.set(config["dynamic_flow"])

	#PARTICLES
	particles = config["particles"]

	#BOUND CONDS.
	#Flow
	flux.set(config["flux"]) # what is this?
	flux_type.set(config["flux_type"]) # what is this?
	is_dynamic_flow_velocity.set(config["is_dynamic_flow_velocity"] )

	project_name.set(config["project_name"])
def collect_data(save=True):
	config = {}

	#HYDRO
	#Main Hydro
	config["celerity_cm_min"] = bedform_celerity_v.get()
	config["flow_vel"] = flow_vel_v.get()
	config["depth_of_sediment"] = depth_of_sediment_v.get()
	config["specific_storage"] = specific_storage_v.get()
	config["water_depth_cm"] = water_depth_v.get()
	config["porosity"] = porosity_v.get()

	#Conductivity
	config["Ks_sand"] = global_k_v.get()
	config['feedback'] = is_clogging.get()
	config["clogging_type"] = clogging_type.get()

	#Bedform Shape
	config["tri_wavelength"] = tri_wavelength.get()
	config['tri_height'] = tri_height.get()
	config['tri_crest'] = tri_crest.get()

	config["x_min"] = 0
	config["x_max"] = domain_length.get()
	config["J"]= domain_nodesx.get() # note: as the model is written, J actually refers to the number of segments in the x-direction, so the number of x nodes is J+1
	config["dx"] = domain_dx.get()

	config["z_min"] = 0
	config["z_max"] = domain_height.get()
	config["I"] = domain_nodesz.get() # same comment as for J above
	config["dz"] = domain_dz.get()

	config["domain_width"] = domain_width.get()
	config["bedform_dx"] = bedform_dx.get()
	#Time
	config["sim_duration_min"] = sim_duration.get()
	config['dt'] = domain_dt.get()
	config["displacement_per_timestep_cm"] = domain_dx.get()

	#Animation
	config["frame_duration"] = frame_duration.get()
	config["head_margin_cm"] = head_margin.get() # since head behaves anomalously near side boundary, leave a margin at either side boundary when doing particle tracking calculations

	#Extras
	config["include_surface_water_conc"] = include_surface_water_conc.get()
	config["dynamic_flow"] = dynamic_flow.get()

	#PARTICLES
	config["particles"] = particles
	config["particle_specs"] = []
	# 	config["particle_specs"] = [{Simulation.KEY_FC: 0.9, Simulation.KEY_SV: 0}]

	#BOUND CONDS.
	#Flow
	config["flux"] = flux.get() # what is this?
	config["flux_type"] = flux_type.get() # what is this?
	config["is_dynamic_flow_velocity"] = is_dynamic_flow_velocity.get()
	print(config)


	# the ratio of how much simulation time should pass per unit of real time
	# for example, if I set this value to 180, it means that I want 180 minutes of simulated time to pass in 1 minute of real time
	# used in generating the animation of a simulation
	config["sim_time_ratio"] = 180

	# bedform shape properties
	config["wavelength_cm"] = 25
	config["bedform_height_cm"] = 2.5
	config["crest_offset_cm"] = 22

	config["flume_width_cm"] = 30

	config["a_L"] = 10
	config["a_T"] = config["a_L"] * 0.1

	config["project_name"] = project_name.get()

	if save:
		with open('obj/'+ project_name.get() + '.pkl', 'wb') as f:
			pickle.dump(config, f, pickle.HIGHEST_PROTOCOL)

	s = Simulation.from_object(config)
	s.compute_initial_conditions()
	fig = s.plot_initial_conditions()

	return fig
def run_hydro_window():
	global hydrological, project_win, bound_window, domain_window, animation_window, particle_win
	hydro_button.config(bg="blue");project_button.config(bg="green");simulation_button.config(bg="green");animation_button.config(bg="green");domain_button.config(bg="green");particles_button.config(bg="green")

	try:
		bound_window.destroy()
	except:
		pass
	try:
		domain_window.destroy()
	except:
		pass
	try:
		animation_window.destroy()
	except:
		pass
	try:
		particle_win.destroy()
	except:
		pass
	try:
		pass
		hydrological.destroy()
	except:
		pass

	hydrological = Toplevel()
	hydrological.geometry("550x650")
	hydrological.title("Hydrological")

	def grey_global_k():

		try:
			#global_k_x_v.set(global_k_v.get())
			#global_k_y_v.set(global_k_v.get())
			pass
		except:
			pass

		#global_k_v.set("")

		global_k_e.config(state='disable')
		global_hvy_e.config(state='normal')
		global_hvx_e.config(state='normal')

	def grey_global_hv():

		#global_k_x_v.set("")
		#global_k_y_v.set("")
		global_k_e.config(state='normal')
		global_hvy_e.config(state='disable')
		global_hvx_e.config(state='disable')

	def enable_clogging_relation():
		if is_clogging.get():
			clogging_menu.configure(state ='normal')
		else:
			#clogging_type.set("None")
			clogging_menu.configure(state ='disabled')

	def grey_custom_shape():
		wavelength_e.config(state='normal')
		height_e.config(state='normal')
		crest_e.config(state='normal')

	def grey_triangle_shape():
		#tri_wavelength.set("")
		#tri_height.set("")
		#tri_crest.set("")
		wavelength_e.config(state='disable')
		height_e.config(state='disable')
		crest_e.config(state='disable')


	def draw_triangle(*args):
		try:
			mycanvas.delete('all')
			START_HEIGHT = 70
			SCALE_C = 7
			MID = 150
			h = START_HEIGHT - tri_height.get()*SCALE_C
			point_a = (MID-tri_wavelength.get()*SCALE_C/2, START_HEIGHT)
			point_b = (MID+tri_wavelength.get()*SCALE_C/2, START_HEIGHT)
			point_c = (MID+tri_crest.get()*SCALE_C/2-tri_wavelength.get()*SCALE_C/2/2,h)
			mycanvas.create_line(*point_a, *point_b,)
			mycanvas.create_line(*point_a, *point_c)
			mycanvas.create_line(*point_b, *point_c)
		except Exception as e:
			#print(e)
			pass



	#FRAMES
	hydro1 = LabelFrame(hydrological, text="Main Hydro", padx=55,pady=5)
	hydro1.place(x=21,y=10)

	hydro2 = LabelFrame(hydrological, text="Hydraulic Conductivity", padx=5,pady=10)
	hydro2.place(x=20,y=180)

	hydro3 = LabelFrame(hydrological, text="Bedform Shape", padx= 40,pady=10)
	hydro3.place(x=20,y=330)


	#LABELS:
	bedform_celerity_l = Label(hydro1, padx=100, text="Bedform Celerity (cm/hr) ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=0)
	flow_vel_l = Label(hydro1, padx=100,text="Flow Velocity (m/s)         ", fg="black",justify="left", font=("arial", 10, "bold")).grid(column = 0,row=1)
	depth_of_sediment_l = Label(hydro1, padx=100, text="Depth Of Sediment (cm) ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=2)
	specific_storage_l = Label(hydro1, padx=100, text="Specific Storage (-)         ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=3)
	water_depth_l = Label(hydro1, padx=100, text="Water Depth (cm)           ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=4)
	porosity_l = Label(hydro1, padx=100, text="Porosity                      ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=5)

	tri_wave_l = Label(hydro3, padx=100, text="Wavelength ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 1,row=1)
	tri_height_l = Label(hydro3, padx=100, text="Height ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 1,row=2)
	tri_crest_l = Label(hydro3, padx=100, text="Crest Offset ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 1,row=3)



	#ENTRIES
	bedform_celerity_e = Entry(hydro1, textvar=bedform_celerity_v, width = 5)
	bedform_celerity_e.grid(column = 1,row=0)

	flow_vel_e = Entry(hydro1, textvar=flow_vel_v, width = 5)
	flow_vel_e.grid(column = 1,row=1)

	depth_of_sediment_e = Entry(hydro1, textvar=depth_of_sediment_v, width = 5)
	depth_of_sediment_e.grid(column = 1,row=2)

	specific_storage_e = Entry(hydro1, textvar=specific_storage_v, width = 5)
	specific_storage_e.grid(column = 1,row=3)

	water_depth_e = Entry(hydro1, textvar=water_depth_v, width = 5)
	water_depth_e.grid(column = 1,row=4)

	porosity_e = Entry(hydro1, textvar=porosity_v, width = 5)
	porosity_e.grid(column = 1,row=5)

	global_k_e = Entry(hydro2, textvar=global_k_v, width = 5)
	global_k_e.grid(column = 1,row=0)

	global_hvx_e = Entry(hydro2, textvar=global_k_x_v, width = 5,state ='disabled')
	global_hvx_e.grid(column = 1,row=1)

	global_hvy_e = Entry(hydro2, textvar=global_k_y_v, width = 5,state ='disabled')
	global_hvy_e.grid(column = 2,row=1)

	clogging_menu = OptionMenu(hydro2, clogging_type, "one", "two", "three")
	clogging_menu.configure(state ='disabled')
	clogging_menu.grid(column = 1,row=3)

	Radiobutton(hydro2, text="Global K",padx = 150, font=("arial", 10, "bold"), variable = radio_for_global,  command=grey_global_hv, value=0).grid(column = 0,row=0)
	Radiobutton(hydro2, text="Global H/V",padx = 20, font=("arial", 10, "bold"), variable = radio_for_global, command=grey_global_k, value=1).grid(column = 0,row=1)
	Radiobutton(hydro2, text="Custom",padx = 20, font=("arial", 10, "bold"), variable = radio_for_global, command=ps, value=2).grid(column = 0,row=2)

	define_button = Button(hydro2, text="Define",height=1, width=5, bg="light grey", fg="black",command=ps)
	define_button.grid(column = 1,row=2)

	button1_ttp = CreateToolTip(define_button, "mouse is over button 1")
	Checkbutton(hydro2, text='Feedback (Clogging)', font=("arial", 10, "bold"), variable=is_clogging, onvalue=True, offvalue=False, command=enable_clogging_relation).grid(column = 0,row=3)

	Radiobutton(hydro3, text="Triangular",padx = 10, font=("arial", 10, "bold"), variable = radio_for_tri, command=grey_custom_shape, value=0).grid(column = 0,row=0)
	Radiobutton(hydro3, text="Custom",padx = 10, font=("arial", 10, "bold"), variable = radio_for_tri, command=grey_triangle_shape, value=1).grid(column = 0,row=5)



	wavelength_e = Entry(hydro3, textvar=tri_wavelength, width = 5,state ='normal')
	wavelength_e.grid(column = 2,row=1)
	tri_wavelength.trace("w", draw_triangle)

	height_e = Entry(hydro3, textvar=tri_height, width = 5,state ='normal')
	height_e.grid(column = 2,row=2)
	tri_height.trace("w", draw_triangle)

	crest_e = Entry(hydro3, textvar=tri_crest, width = 5,state ='normal')
	crest_e.grid(column = 2,row=3)
	tri_crest.trace("w", draw_triangle)

	#triangle_button = Button(hydro3, text="Triangle",height=1, width=5, bg="light grey", fg="black",command=draw_triangle)
	#triangle_button.grid(column = 2,row=4)

	mycanvas = Canvas(hydro3, width=300, height=100, bg='white')
	mycanvas.grid(column = 1,row=6)


	define_button2 = Button(hydro3, text="Define",height=1, width=5, bg="light grey", fg="black",command=ps)
	define_button2.grid(column = 2,row=5)

	draw_triangle()
def run_project_window():
	global project_win
	hydro_button.config(bg="green");project_button.config(bg="blue");simulation_button.config(bg="green");animation_button.config(bg="green");domain_button.config(bg="green");particles_button.config(bg="green")

	try:
		project_win.destroy()
	except:
		pass
	
	project_win = Toplevel()
	project_win.geometry("650x350")
	project_win.title("Project")

	Label(project_win, padx=0, text="Project:",justify="left", fg="black", font=("arial", 25, "bold")).grid(column = 1,row=0)
	
	Radiobutton(project_win, text="New",padx = 10, font=("arial", 10, "bold"), variable = project_type, command=ps, value=0).grid(column = 0,row=1)
	Radiobutton(project_win, text="Import Settings From Existing Project",padx = 10, font=("arial", 10, "bold"), variable = project_type, command=ps, value=1).grid(column = 0,row=3)
	Radiobutton(project_win, text="Resume Existing Project",padx = 10, font=("arial", 10, "bold"), variable = project_type, command=ps, value=2).grid(column = 0,row=6)

	Entry(project_win, textvar=project_name, width = 15,state ='normal').grid(column = 1,row=2)
	Entry(project_win, textvar=project_name, width = 15,state ='normal').grid(column = 1,row=4)
	Entry(project_win, textvar=project_name, width = 15,state ='normal').grid(column = 1,row=7)

	button_explore = Button(project_win,text = "Browse Files", command = open_text_file)
	button_explore.grid(column=1,row=5)
 
	button_explore = Button(project_win,text = "Browse Files", command = open_text_file)
	button_explore.grid(column=1,row=8)
def run_particle_window():
	global hydrological, project_win, bound_window, domain_window, animation_window, particle_win,labels
	hydro_button.config(bg="green");project_button.config(bg="green");simulation_button.config(bg="green");animation_button.config(bg="green");domain_button.config(bg="green");particles_button.config(bg="blue")


	try:
		bound_window.destroy()
	except:
		pass
	try:
		domain_window.destroy()
	except:
		pass
	try:
		animation_window.destroy()
	except:
		pass
	try:
		particle_win.destroy()
	except:
		pass
	try:
		pass
		hydrological.destroy()
	except:
		pass
	particle_win = Toplevel()
	particle_win.geometry("850x650")
	particle_win.title("Particles")

	particle1 = LabelFrame(particle_win, text="Particles", padx=0,pady=5)
	particle1.grid(column=0,row=0)

	buttonframe = LabelFrame(particle_win, text="Particles", padx=0,pady=5)
	buttonframe.grid(column=0,row=1)
	frames = [LabelFrame(particle_win, text="Particles", padx=0,pady=5) for _ in range (10)]
	#for i, frame in enumerate(frames):
	#	frame.place
	labels = []
	Label(particle1, padx=0, text="Particles:", font=("arial", 20, "bold")).grid(column=0,row=0)
	for i in range(1, 10):
		Label(particle1, padx=0, text=str(i)+'.', font=("arial", 20, "bold")).grid(column=0,row=i)
	OptionMenu(buttonframe,to_del, 1,2,3,4,5,6,7,8,9).grid(column=4,row=1)
	OptionMenu(buttonframe,to_edit, 1,2,3,4,5,6,7,8,9).grid(column=2,row=1)

	options = ['|Name|', "|Filtration Coefficient|", "|Settling Velocity|", "|Diffusion Coefficient|", "|Color In Plots|", "|Particle Value|", "|Leaves Trail?|"]
	for i, o in enumerate(options):
		Label(particle1, padx=0, text=o, font=("arial", 9, "bold")).grid(column=i+1,row = 0)

	def display_particles():
		global particles,labels

		for l in labels:
			for item in l:
				item.destroy()

		labels = []
		for i, p in enumerate(particles):
			l=[]
			things = [p["name"], p["filtration_coefficient"], p['settling_velocity'], p['diffusion_coefficient'],
			p["color_in_plots"], p["particle_value"], p["is_trail"]]

			for j, thing in enumerate(things):
				item = Label(particle1, padx=0, text="|" + str(thing) + "|", font=("arial", 15, "bold"))
				item.grid(column= 1+j,row= i+1)
				l.append(item)
			labels.append(l)

	def add_particle():

		def cancel():
			addpart_window.destroy()
		def confirm():
			global particles, particle_menu

			new_particle = {}
			new_particle['name'] = particle_name_v.get()
			new_particle["filtration_coefficient"] = filtration_c_v.get()
			new_particle["settling_velocity"] = settling_vel_v.get()
			new_particle["diffusion_coefficient"] = diffusion_c_v.get()
			new_particle["color_in_plots"] = color_in_plots.get()
			new_particle["particle_value"]= particle_value.get()
			new_particle["is_trail"]=leave_trail.get()
			new_particle["sw_conc"] = ''
			new_particle["pumping"] = False
			new_particle["turnover"] = False


			particles.append(new_particle)
			print(particles, '\n\n')

			display_particles()
			addpart_window.destroy()


		addpart_window = Toplevel()
		addpart_window.geometry("700x400")
		addpart_window.title("New Particle")

		addparticle1 = LabelFrame(addpart_window, text="Particles", padx=150,pady=5)
		addparticle1.place(x=21,y=10)

		Label(addparticle1, padx=50, text="Name:                       ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=0)
		Label(addparticle1, padx=50, text="Settling Velocity:       ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=1)
		Label(addparticle1, padx=50, text="Filtration Coefficient:",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=2)
		Label(addparticle1, padx=50, text="Diffusion Coefficient: ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=3)
		Label(addparticle1, padx=10, text="Color In Plots:           ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=4)
		Label(addparticle1, padx=50, text="Particle Weighting:   ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=5)
		Label(addparticle1, padx=10, text="Particle Value:",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=6)
		#Label(addparticle1, padx=10, text="Leave Trail?:  ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=7)

		name_e = Entry(addparticle1, textvar=particle_name_v, width = 5,state ='normal')
		name_e.grid(column = 1,row=0)

		setteling_v_e = Entry(addparticle1, textvar=settling_vel_v, width = 5,state ='normal')
		setteling_v_e.grid(column = 1,row=1)

		filtration_c_e = Entry(addparticle1, textvar=filtration_c_v, width = 5,state ='normal')
		filtration_c_e.grid(column = 1,row=2)

		diffusion_c_e = Entry(addparticle1, textvar=diffusion_c_v, width = 5,state ='normal')
		diffusion_c_e.grid(column = 1,row=3)
		####
		#namelist = [x.name for x in particles]
		#namelist = [1,2,3]

		color_list = ['Red', 'Green', 'Blue']
		color_menu = OptionMenu(addparticle1, color_in_plots, *color_list)
		color_menu.grid(column = 1,row=4)

		particle_value_e = Entry(addparticle1, textvar=particle_value, width = 5,state ='normal')
		particle_value_e.grid(column = 2,row=6)

		Checkbutton(addparticle1, text='Leave Trail?', font=("arial", 10, "bold"), variable=leave_trail, onvalue=True, offvalue=False).grid(column = 0,row=7)

		cncl = Button(addparticle1, text="Cancel",height=1, width=2, bg="red", fg="black",command=cancel,padx=20).grid(column = 3,row=7)
		ok = Button(addparticle1, text="OK",height=1, width=2, bg="red", fg="black",command=confirm,padx=20).grid(column = 4,row=7)

	def delete_particle():
		global particles
		try:
			if len(particles) >= to_del.get():
				print(to_del.get())
				particles.pop(to_del.get()-1)
				display_particles()
		except Exception as e:
			print(e)

	def edit_particle():
		global particles
		if len(particles) >= to_edit.get():
			def cancel():
				editpart_window.destroy()
			def confirm():
				global particles

				particles[to_edit.get()-1]['name'] = particle_name_v.get()
				particles[to_edit.get()-1]["filtration_coefficient"] = filtration_c_v.get()
				particles[to_edit.get()-1]["settling_velocity"] = settling_vel_v.get()
				particles[to_edit.get()-1]["diffusion_coefficient"] = diffusion_c_v.get()
				particles[to_edit.get()-1]["color_in_plots"] = color_in_plots.get()
				particles[to_edit.get()-1]["particle_value"]= particle_value.get()
				particles[to_edit.get()-1]["is_trail"]=leave_trail.get()

				display_particles()
				editpart_window.destroy()


			editpart_window = Toplevel()
			editpart_window.geometry("700x400")
			editpart_window.title("Edit Particle")

			editpart1 = LabelFrame(editpart_window, text="Particles", padx=150,pady=5)
			editpart1.place(x=21,y=10)

			Label(editpart1, padx=50, text="Name:                      ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=0)
			Label(editpart1, padx=50, text="Settling Velocity:     ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=1)
			Label(editpart1, padx=50, text="Filtration Coefficient:",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=2)
			Label(editpart1, padx=50, text="Diffusion Coefficient: ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=3)
			Label(editpart1, padx=10, text="Color In Plots:        ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=4)
			Label(editpart1, padx=50, text="Particle Weighting:   ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=5)
			Label(editpart1, padx=10, text="Particle Value:",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=6)
			#Label(addparticle1, padx=10, text="Leave Trail?:  ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=7)

			particle_name_v.set(particles[to_edit.get()-1]['name'])
			name_e = Entry(editpart1, textvar=particle_name_v, width = 5,state ='normal')
			name_e.grid(column = 1,row=0)

			settling_vel_v.set(particles[to_edit.get()-1]['settling_velocity'])
			setteling_v_e = Entry(editpart1, textvar=settling_vel_v, width = 5,state ='normal')
			setteling_v_e.grid(column = 1,row=1)

			filtration_c_v.set(particles[to_edit.get()-1]['filtration_coefficient'])
			filtration_c_e = Entry(editpart1, textvar=filtration_c_v, width = 5,state ='normal')
			filtration_c_e.grid(column = 1,row=2)

			diffusion_c_v.set(particles[to_edit.get()-1]['diffusion_coefficient'])
			diffusion_c_e = Entry(editpart1, textvar=diffusion_c_v, width = 5,state ='normal')
			diffusion_c_e.grid(column = 1,row=3)
			####
			#namelist = [x.name for x in particles]
			#namelist = [1,2,3]

			color_in_plots.set(particles[to_edit.get()-1]['color_in_plots'])
			color_list = ['Red', 'Green', 'Blue']
			color_menu = OptionMenu(editpart1, color_in_plots, *color_list)
			color_menu.grid(column = 1,row=4)

			particle_value.set(particles[to_edit.get()-1]['particle_value'])
			particle_value_e = Entry(editpart1, textvar=particle_value, width = 5,state ='normal')
			particle_value_e.grid(column = 2,row=6)

			leave_trail.set(particles[to_edit.get()-1]['is_trail'])
			Checkbutton(editpart1, text='Leave Trail?', font=("arial", 10, "bold"), variable=leave_trail, onvalue=True, offvalue=False).grid(column = 0,row=7)

			cncl = Button(editpart1, text="Cancel",height=1, width=2, bg="red", fg="black",command=cancel,padx=20).grid(column = 3,row=7)
			ok = Button(editpart1, text="OK",height=1, width=2, bg="red", fg="black",command=confirm,padx=20).grid(column = 4,row=7)
		#Button(particle1, text="",height=1, width=2, bg="light grey", fg="black",command=ps,padx=20,state="disabled").grid(column = 1,row=0)
		#Button(particle1, text="",height=1, width=2, bg="light grey", fg="black",command=ps,padx=20,state="disabled").grid(column = 3,row=0)

	add = Button(buttonframe, text="Add",height=1, width=2, bg="light blue", fg="black",command=add_particle,padx=20)
	add.grid(column = 0,row=0)

	
	edit = Button(buttonframe, text="Edit",height=1, width=2, bg="light blue", fg="black",command=edit_particle,padx=20)
	edit.grid(column = 2,row=0)
	
	delete = Button(buttonframe, text="Delete",height=1, width=2, bg="red", fg="black",command=delete_particle,padx=20)
	delete.grid(column = 4,row=0)
	display_particles()
def run_bound_window():
	global hydrological, project_win, bound_window, domain_window, animation_window, particle_win, particles
	hydro_button.config(bg="green");project_button.config(bg="green");simulation_button.config(bg="blue");animation_button.config(bg="green");domain_button.config(bg="green");particles_button.config(bg="green")

	try:
		bound_window.destroy()
	except:
		pass
	try:
		domain_window.destroy()
	except:
		pass
	try:
		animation_window.destroy()
	except:
		pass
	try:
		particle_win.destroy()
	except:
		pass
	try:
		pass
		hydrological.destroy()
	except:
		pass
	bound_window = Toplevel()
	bound_window.geometry("550x650")
	bound_window.title("Boundary Conditions")
	
	
	def enable_constant_flux():
		flux_e.config(state = "normal")
		flux_m.configure(state = "normal")
		define_button1.config(state="disabled")
	def grey_constant_flux():
		flux_e.config(state = "disabled")
		flux_m.configure(state = "disabled")
		define_button1.config(state="normal")

	def disable_button_flow_vel():
		if is_dynamic_flow_velocity.get():
			define_button2.config(state = "normal")
		else:
			define_button2.config(state = "disabled")


	def particle_bs():

		def pump_particle():

			name = listbox.get(ANCHOR)
			for particle in particles:
				if particle["name"] == name:
					if particle["pumping"] == True:
						particle["pumping"] = False
					else:
						particle["pumping"] = True
			print(name, "pump")
			bound_window.destroy()
			run_bound_window()
		
		def turn_particle():

			name = listbox.get(ANCHOR)
			for particle in particles:
				if particle["name"] == name:
					if particle["turnover"] == True:
						particle["turnover"] = False
					else:
						particle["turnover"] = True
			print(name, "turnover")
			bound_window.destroy()
			run_bound_window()
		particle_bs_menu = Toplevel()
		particle_bs_menu.geometry("700x400+500+30")
		particle_bs_menu.title("Particle Boundary Conditions")

		bound1 = LabelFrame(particle_bs_menu, text="", padx=0,pady=5)
		bound1.grid(column=0,row=0)

		bound2 = LabelFrame(particle_bs_menu, text="", padx=0,pady=5)
		bound2.grid(column=0,row=1)

		bound3 = LabelFrame(particle_bs_menu, text="", padx=0,pady=5)
		bound3.grid(column=0,row=2)

		Label(bound1, padx=0, text="Particle Boundary Conditions:", fg="black", font=("arial", 25, "bold")).grid(column = 0,row=0)

		names = [particle["name"] for particle in particles]
		#names = ['uri', 'shahar', 'namoni']

		listbox = Listbox(bound2)
		listbox.grid(column=0,row=1)
		for name in names:
			listbox.insert(END, name)
		
		pumping_button = Button(bound3, text="Pumping",height=1, width=8, bg="green", fg="black",command=pump_particle, font=("arial", 10, "bold"))
		pumping_button.grid(column = 0,row=0)

		turnover_button = Button(bound3, text="Turnover",height=1, width=8, bg="green", fg="black",command=turn_particle, font=("arial", 10, "bold"))
		turnover_button.grid(column = 1,row=0)

		'''
		for i, particle in enumerate(particles):
			name = particle["name"]
			frame = LabelFrame(bound2, text="", padx=0,pady=5)
			frame.grid(column=0, row=i)
			Label(frame, padx=0, text=name, fg="black", font=("arial", 25, "bold")).grid(column = 0,row=0)
			Entry(frame, textvar=flux, width = 5,state ='normal').grid(column = 1,row=0)
			Checkbutton(frame, text='Pumping', font=("arial", 10, "bold"), command = pump_particle).grid(column = 2,row=0)
			Checkbutton(frame, text='Turnover', font=("arial", 10, "bold"), command = turn_particle).grid(column = 3,row=0)
		'''



	bound1 = LabelFrame(bound_window, text="Flow", padx=0,pady=5)
	bound1.place(x=21,y=10)

	bound2 = LabelFrame(bound_window, text="Particles", padx=0,pady=5)
	bound2.place(x=21,y=250)

	Label(bound1, padx=0, text="Flow:",justify="left", fg="black", font=("arial", 25, "bold")).grid(column = 0,row=0)
	Label(bound1, padx=10, text="Losing/Gaining Flux			 ",justify="left", fg="black", font=("arial",14, "bold")).grid(column = 1,row=1)

	Radiobutton(bound1, text="Constant",padx = 10, font=("arial", 10, "bold"), variable = radio_for_flux, command=enable_constant_flux, value=0).grid(column = 1,row=2)
	Radiobutton(bound1, text="Custom",padx = 10, font=("arial", 10, "bold"), variable = radio_for_flux, command=grey_constant_flux, value=1).grid(column = 1,row=3)

	flux_e = Entry(bound1, textvar=flux, width = 5,state ='normal')
	flux_e.grid(column = 2,row=2)

	flux_m = OptionMenu(bound1, flux_type, "cm/day", "two", "three")
	flux_m.grid(column = 4,row=2)

	define_button1 = Button(bound1, text="Define",height=1, width=5, bg="light grey", fg="black",command=ps,state = "disabled")
	define_button1.grid(column = 2,row=3)

	Checkbutton(bound1, text='Dynamic Flow Velocity', font=("arial", 10, "bold"), variable=is_dynamic_flow_velocity, onvalue=True, offvalue=False,command = disable_button_flow_vel).grid(column = 1,row=4)

	define_button2 = Button(bound1, text="Define",height=1, width=5, bg="light grey", fg="black",command=ps,state = "disabled")
	define_button2.grid(column = 1,row=5)


	Label(bound2, padx=10, text="Particles:",justify="left", fg="black", font=("arial", 25, "bold")).grid(column = 0,row=0)

	for i, particle in enumerate(particles):
		Label(bound2, padx=10, text=particle["name"] ,justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=i+1)
		if particle['pumping'] == True:
			Label(bound2, padx=10, text="Pumping" ,justify="left", fg="green", font=("arial", 10, "bold")).grid(column = 1,row=i+1)
		else:
			Label(bound2, padx=10, text="Pumping" ,justify="left", fg="red", font=("arial", 10, "bold")).grid(column = 1,row=i+1)

		if particle['turnover'] == True:
			Label(bound2, padx=10, text="Turnover" ,justify="left", fg="green", font=("arial", 10, "bold")).grid(column = 2,row=i+1)
		else:
			Label(bound2, padx=10, text="Turnover" ,justify="left", fg="red", font=("arial", 10, "bold")).grid(column = 2,row=i+1)


	edit_particles_button = Button(bound2, text="Edit Particle BCs",height=1, width=15, bg="light grey", fg="black",command=particle_bs)
	edit_particles_button.grid(column = 5,row=10)
def run_domain_window():
	global hydrological, project_win, bound_window, domain_window, animation_window, particle_win
	hydro_button.config(bg="green");project_button.config(bg="green");simulation_button.config(bg="green");animation_button.config(bg="green");domain_button.config(bg="blue");particles_button.config(bg="green")

	try:
		bound_window.destroy()
	except:
		pass
	try:
		domain_window.destroy()
	except:
		pass
	try:
		animation_window.destroy()
	except:
		pass
	try:
		particle_win.destroy()
	except:
		pass
	try:
		pass
		hydrological.destroy()
	except:
		pass
	domain_window = Toplevel()
	domain_window.geometry("550x650")
	domain_window.title("Model And Simulation")

	def update_dx(*args):
		try:
			domain_dx.set(domain_length.get()/domain_nodesx.get())
		except:
			pass

		try:
			domain_dz.set(domain_height.get()/domain_nodesz.get())
		except:
			pass

	def draw_domain(*args):
		try:
			fig = collect_data(save=False)

			canvas = FigureCanvasTkAgg(fig, master=domain5)
			canvas.get_tk_widget().grid(row=1,column=24)
			canvas.draw()
		except:
			pass
	domain1 = LabelFrame(domain_window, text="Domain", padx=0,pady=5)
	domain1.place(x=21,y=10)

	domain2 = LabelFrame(domain_window, text="Time", padx=0,pady=5)
	domain2.place(x=21,y=120)

	domain3 = LabelFrame(domain_window, text="Animation", padx=0,pady=5)
	domain3.place(x=21,y=230)

	domain4 = LabelFrame(domain_window, text="Extras", padx=0,pady=5)
	domain4.place(x=21,y=310)

	domain5 = LabelFrame(domain_window, text="Domain Simulation", padx=0,pady=5)
	domain5.place(x=21,y=410)

	Label(domain1, padx=10, text="Length (cm)     ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=0)
	Label(domain1, padx=10, text="# Segments X     ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=1)
	Label(domain1, padx=10, text="dx (cm         ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=2)
	Label(domain1, padx=10, text="Height (cm)          ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 2,row=0)
	Label(domain1, padx=10, text="# Segments Z     ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 2,row=1)
	Label(domain1, padx=10, text="dz (cm)         ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 2,row=2)
	Label(domain1, padx=10, text="Width (cm)           ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 4,row=0)
	Label(domain1, padx=10, text="Water Depth (cm)",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 4,row=1)
	Label(domain1, padx=10, text="Sediment Depth  ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 4,row=2)

	length_e = Entry(domain1, textvar=domain_length, width = 5,state ='normal')
	length_e.grid(column = 1,row=0)
	domain_length.trace("w", update_dx)
	domain_length.trace("w", draw_domain)

	nodesx_e = Entry(domain1, textvar=domain_nodesx, width = 5,state ='normal')
	nodesx_e.grid(column = 1,row=1)
	domain_nodesx.trace("w", update_dx)


	dx_e = Entry(domain1, textvar=domain_dx, width = 5,state ='disabled')
	dx_e.grid(column = 1,row=2)

	height_e = Entry(domain1, textvar=domain_height, width = 5,state ='normal')
	height_e.grid(column = 3,row=0)
	domain_height.trace("w", update_dx)
	domain_height.trace("w", draw_domain)

	nodesz_e = Entry(domain1, textvar=domain_nodesz, width = 5,state ='normal')
	nodesz_e.grid(column = 3,row=1)
	domain_nodesz.trace("w", update_dx)

	dz_e = Entry(domain1, textvar=domain_dz, width = 5,state ='disabled')
	dz_e.grid(column = 3,row=2)

	width_e = Entry(domain1, textvar=domain_width, width = 5,state ='normal')
	width_e.grid(column = 5,row=0)

	seidment_depth_e = Entry(domain1, textvar=depth_of_sediment_v, width = 5,state ='normal')
	seidment_depth_e.grid(column = 5,row=2)

	water_depth_e = Entry(domain1, textvar=water_depth_v, width = 5,state ='normal')
	water_depth_e.grid(column = 5,row=1)

	Label(domain2, padx=130, text="Simulation Duration (min) ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=0)
	Label(domain2, padx=130, text="dt (s)                                 ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=1)
	Label(domain2, padx=130, text="Bedform dx (cm/min)        ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=2)

	sim_dur_e = Entry(domain2, textvar=sim_duration, width = 5,state ='normal')
	sim_dur_e.grid(column = 1,row=0)

	dt_e = Entry(domain2, textvar=domain_dt, width = 5,state ='normal')
	dt_e.grid(column = 1,row=1)

	bedform_dx_e = Entry(domain2, textvar=bedform_dx, width = 5,state ='normal')
	bedform_dx_e.grid(column = 1,row=2)

	Label(domain3, padx=150, text="Frame Duration (ms)",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=0)
	Label(domain3, padx=150, text="Head Margin (cm)    ",justify="left", fg="black", font=("arial", 10, "bold")).grid(column = 0,row=1)

	frame_dur_e = Entry(domain3, textvar=frame_duration, width = 5,state ='normal')
	frame_dur_e.grid(column = 1,row=0)


	head_margin_e = Entry(domain3, textvar=head_margin, width = 5,state ='normal')
	head_margin_e.grid(column = 1,row=1)

	Checkbutton(domain4, text='Include surface water conc.', font=("arial", 10, "bold"), variable=include_surface_water_conc, onvalue=True, offvalue=False).grid(column = 0,row=0)
	Checkbutton(domain4, text='Dynamic flow                    ', font=("arial", 10, "bold"), variable=dynamic_flow, onvalue=True, offvalue=False).grid(column = 0,row=1)

	draw_domain()
def run_animation_window():
	global hydrological, project_win, bound_window, domain_window, animation_window, particle_win
	hydro_button.config(bg="green");project_button.config(bg="green");simulation_button.config(bg="green");animation_button.config(bg="blue");domain_button.config(bg="green");particles_button.config(bg="green")

	try:
		bound_window.destroy()
	except:
		pass
	try:
		domain_window.destroy()
	except:
		pass
	try:
		animation_window.destroy()
	except:
		pass
	try:
		particle_win.destroy()
	except:
		pass
	try:
		pass
		hydrological.destroy()
	except:
		pass
	animation_window = Toplevel()
	animation_window.geometry("550x650")
	animation_window.title("Animation")
#############################BUTTONS FOR NEW WINDOWS

Entry(root, textvar=project_name, width=12).place(x=10,y=130)


hydro_button = Button(root, text="Hydro", width=12, bg="green", fg="white",command=run_hydro_window)
hydro_button.grid(column = 0,row=1)

particles_button = Button(root, text="Particles", width=12, bg="green", fg="white",command=run_particle_window)
particles_button.grid(column = 0,row=2)

simulation_button = Button(root, text="Boundary Conds.", width=12, bg="green", fg="white",command=run_bound_window)
simulation_button.grid(column = 0,row=3)

domain_button = Button(root, text="Model and Sim.", width=12, bg="green", fg="white",command=run_domain_window)
domain_button.grid(column = 0,row=4)

animation_button = Button(root, text="Animation", width=12, bg="green", fg="white",command=run_animation_window)
#animation_button.grid(column = 0,row=5)

run_button = Button(root, text="Run", width=12, bg="green", fg="white",command=collect_data)
run_button.place(x=0,y=185)

project_button = Button(root, text="Import Project", width=12, bg="green", fg="white",command=open_text_file)

import_project = Button(root, text="Import Project", width=12, bg="light grey", fg="black",command=open_text_file)
import_project.place(x=0,y=160)

run_bound_window()
root.mainloop()

'''
###############################
bedform_celerity_v = DoubleVar()
flow_vel_v = DoubleVar()
depth_of_sediment_v = DoubleVar()
specific_storage_v = DoubleVar()
water_depth_v = DoubleVar()
porosity_v = DoubleVar()
################################

################################
global_k_v = DoubleVar()

global_k_x_v = DoubleVar()
global_k_y_v = DoubleVar()

is_clogging = BooleanVar()
clogging_type = StringVar()
################################

################################
tri_wavelength = DoubleVar()
tri_height = DoubleVar()
tri_crest = DoubleVar()
################################
'''