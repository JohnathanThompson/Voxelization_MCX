import sv
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET
from xml.dom import minidom

# -------------------------------------------------------
# STEP 1 — Generate Mesh (UNCHANGED - known working)
# -------------------------------------------------------
print("=== STEP 1: Meshing ===")

mesher = sv.meshing.TetGen()
mesher.load_model("/usr/games/JayTee/DemoProject/SVProject/Models/demo.vtp")
mesher.compute_model_boundary_faces(angle=50.0)

# Print faces so we can confirm IDs
face_ids = mesher.get_model_face_ids()
face_info = mesher.get_model_face_info()
print("Face IDs:", face_ids)
print("Face Info:", face_info)

# Set wall face — largest face by point count
mesher.set_walls([1])

options = sv.meshing.TetGenOptions(
    global_edge_size=0.4,
    surface_mesh_flag=True,
    volume_mesh_flag=True
)
mesher.generate_mesh(options)

mesh = mesher.get_mesh()

# Write mesh-complete folder
mesher.write_mesh("/home/john_thompson/projects/DemoProject/SVProject/Simulations/demojob/mesh-complete/testmesh.vtp")
print("Mesh written.")

# Print generated surface files
surfaces_dir = "/home/john_thompson/projects/DemoProject/SVProject/Simulations/demojob/mesh-complete/mesh-surfaces/"
print("\nSurface files generated:")
for f in sorted(os.listdir(surfaces_dir)):
    print(f"  {f}")

# -------------------------------------------------------
# STEP 2 — Copy Flow File (UNCHANGED)
# -------------------------------------------------------
print("\n=== STEP 2: Copying flow file ===")

os.makedirs("/home/john_thompson/projects/DemoProject/SVProject/Simulations2/", exist_ok=True)

shutil.copy2(
    "/home/john_thompson/projects/DemoProject/SVProject/flow-files/steady.flow",
    "/home/john_thompson/projects/DemoProject/SVProject/Simulations2/steady.flow"
)
print("Flow file copied.")

# -------------------------------------------------------
# STEP 3 — Write solver.xml
# SPEED IMPROVEMENTS ONLY HERE:
#   - Number_of_time_steps:        200  -> 50
#   - Time_step_size:              0.005 -> 0.01
#   - Increment_in_saving_restart: 10   -> 50
#   - Increment_in_saving_VTK:     10   -> 50
#   - Max_iterations (equation):   12   -> 5
#   - Tolerance (equation):        1e-5 -> 1e-4
#   - LS Max_iterations:           15   -> 10
#   - LS Tolerance:                1e-4 -> 1e-3
#   - Krylov_space_dimension:      200  -> 100
#   - NS_GM_max_iterations:        10   -> 5
#   - NS_GM_tolerance:             1e-3 -> 1e-2
#   - NS_CG_max_iterations:        300  -> 100
#   - NS_CG_tolerance:             1e-3 -> 1e-2
#   - Spatial outputs: 6 fields -> 3 (Pressure, Velocity, WSS only)
# -------------------------------------------------------
print("\n=== STEP 3: Writing solver.xml ===")

root = ET.Element("svMultiPhysicsFile", version="1.0")

gen = ET.SubElement(root, "GeneralSimulationParameters")
ET.SubElement(gen, "Continue_previous_simulation").text = "false"
ET.SubElement(gen, "Number_of_spatial_dimensions").text = "3"
ET.SubElement(gen, "Number_of_time_steps").text = "50"       # was 200
ET.SubElement(gen, "Time_step_size").text = "0.01"            # was 0.005
ET.SubElement(gen, "Spectral_radius_of_infinite_time_step").text = "0.5"
ET.SubElement(gen, "Increment_in_saving_restart_files").text = "50"   # was 10
ET.SubElement(gen, "Start_saving_after_time_step").text = "1"
ET.SubElement(gen, "Save_results_to_VTK_format").text = "true"
ET.SubElement(gen, "Name_prefix_of_saved_VTK_files").text = "result"
ET.SubElement(gen, "Increment_in_saving_VTK_files").text = "50"       # was 10

mesh_el = ET.SubElement(root, "Add_mesh", name="fluid_mesh")
ET.SubElement(mesh_el, "Mesh_file_path").text = f"/home/john_thompson/projects/DemoProject/SVProject/Simulations/demojob/mesh-complete/mesh-complete.mesh.vtu"

face_names = ["cap_aorta", "cap_aorta_2", "cap_right_iliac", "wall_aorta", "wall_right_iliac"]
for face in face_names:
    fe = ET.SubElement(mesh_el, "Add_face", name=face)
    ET.SubElement(fe, "Face_file_path").text = f"/home/john_thompson/projects/DemoProject/SVProject/Simulations/demojob/mesh-complete/mesh-surfaces/{face}.vtp"

eq = ET.SubElement(root, "Add_equation", type="fluid")
ET.SubElement(eq, "Coupled").text = "true"
ET.SubElement(eq, "Min_iterations").text = "3"
ET.SubElement(eq, "Max_iterations").text = "5"        # was 12
ET.SubElement(eq, "Tolerance").text = "1e-4"           # was 1e-5
ET.SubElement(eq, "Backflow_stabilization_coefficient").text = "0.2"
ET.SubElement(eq, "Density").text = "1.06"
visc = ET.SubElement(eq, "Viscosity", model="Constant")
ET.SubElement(visc, "Value").text = "0.04"

ls = ET.SubElement(eq, "LS", type="NS")
la = ET.SubElement(ls, "Linear_algebra", type="fsils")
ET.SubElement(la, "Preconditioner").text = "fsils"
ET.SubElement(ls, "Max_iterations").text = "10"           # was 15
ET.SubElement(ls, "Tolerance").text = "1e-3"               # was 1e-4
ET.SubElement(ls, "Krylov_space_dimension").text = "100"   # was 200
ET.SubElement(ls, "NS_GM_max_iterations").text = "5"       # was 10
ET.SubElement(ls, "NS_GM_tolerance").text = "1e-2"         # was 1e-3
ET.SubElement(ls, "NS_CG_max_iterations").text = "100"     # was 300
ET.SubElement(ls, "NS_CG_tolerance").text = "1e-2"         # was 1e-3

out1 = ET.SubElement(eq, "Output", type="Spatial")
for field in ["Pressure", "Velocity", "WSS"]:              # was 6 fields
    ET.SubElement(out1, field).text = "true"

out2 = ET.SubElement(eq, "Output", type="Boundary_integral")
for field in ["Velocity", "Pressure", "WSS"]:
    ET.SubElement(out2, field).text = "true"

# Inlet BC
bc_in = ET.SubElement(eq, "Add_BC", name="cap_aorta")
ET.SubElement(bc_in, "Type").text = "Dirichlet"
ET.SubElement(bc_in, "Time_dependence").text = "Unsteady"
ET.SubElement(bc_in, "Temporal_values_file_path").text = "/home/john_thompson/projects/DemoProject/SVProject/flow-files/steady.flow"
ET.SubElement(bc_in, "Profile").text = "Parabolic"
ET.SubElement(bc_in, "Impose_flux").text = "true"

# Outlet BCs
for outlet in ["cap_aorta_2", "cap_right_iliac"]:
    bc = ET.SubElement(eq, "Add_BC", name=outlet)
    ET.SubElement(bc, "Type").text = "Neumann"
    ET.SubElement(bc, "Time_dependence").text = "Resistance"
    ET.SubElement(bc, "Value").text = "2000"

# Wall BCs
for wall in ["wall_aorta", "wall_right_iliac"]:
    bc = ET.SubElement(eq, "Add_BC", name=wall)
    ET.SubElement(bc, "Type").text = "Dirichlet"
    ET.SubElement(bc, "Time_dependence").text = "Steady"
    ET.SubElement(bc, "Value").text = "0"

raw = ET.tostring(root, encoding="unicode")
pretty = minidom.parseString(raw).toprettyxml(indent="    ")
lines = pretty.split("\n")[1:]
with open("/home/john_thompson/projects/DemoProject/SVProject/Simulations/solver.xml", "w") as f:
    f.write("\n".join(lines))
print("solver.xml written.")

# -------------------------------------------------------
# STEP 4 — Run Solver (UNCHANGED)
# -------------------------------------------------------
print("\n=== STEP 4: Running solver ===")

result = subprocess.run(
    [
        "mpiexec", "-n", "6",
        "/usr/local/sv/svMultiPhysics/2025-06-20/bin/svmultiphysics",
        "solver.xml"
    ],
    cwd="/home/john_thompson/projects/DemoProject/SVProject/Simulations/",
    capture_output=True,
    text=True
)

print(result.stdout)
if result.returncode != 0:
    print("SOLVER ERROR:")
    print(result.stderr)
else:
    print("Done! Open result_*.vtu files in ParaView.")
