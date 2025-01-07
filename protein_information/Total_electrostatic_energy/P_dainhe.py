import subprocess
import re

def run_apbs(input_file):
    with open("temp_output.txt", "w") as output_file:
        subprocess.run(["apbs", input_file], stdout=output_file)


def extract_total_energy_from_output(output_file):
    with open(output_file, "r") as file:
        lines = file.readlines()
    total_energy = None
    for line in lines:
        if "Total electrostatic energy" in line:
            total_energy = line.split("=")[-1].strip()
            sum= re.findall(r"\d+\.?\d*", total_energy)
            break

    return sum[0]


def total_energy(pro):
    pdb_file = '/home/ntu/Documents/feiailu/MY_Module/protein_information/PDB/Davis_PDB (copy)/' + str(pro) + '.pdb'
    pqr_file = "/home/ntu/Documents/feiailu/MY_Module/protein_information/Total_electrostatic_energy/1XYZ.pqr"
    pdb2pqr_command = f"pdb2pqr --ff=amber --whitespace {pdb_file} {pqr_file}"

    subprocess.run(pdb2pqr_command, shell=True)

    input_file = "/home/ntu/Documents/feiailu/MY_Module/protein_information/Total_electrostatic_energy/input.in"
    with open(input_file, "w") as f:
        f.write("read\n")
        f.write("    mol pqr {}\n".format(pqr_file))
        f.write("end\n\n")
        f.write("elec name com\n")
        f.write("    mg-auto\n")
        f.write("    dime 65 65 65\n")
        f.write("    cglen 81.6 81.6 81.6\n")
        f.write("    fglen 81.6 81.6 81.6\n")
        f.write("    fgcent 41.621 32.573 46.6415\n")
        f.write("    cgcent 41.621 32.573 46.6415\n")
        f.write("    mol 1\n")
        f.write("    npbe\n")
        f.write("    bcfl mdh\n")
        f.write("    srfm smol\n")
        f.write("    chgm spl4\n")
        f.write("    pdie 2.0\n")
        f.write("    sdens 10\n")
        f.write("    sdie 78.54\n")
        f.write("    srad 1.4\n")
        f.write("    temp 300.0\n")
        f.write("    mol 1\n")
        f.write("    calcenergy total\n")

        f.write("    write pot flat /home/ntu/Documents/feiailu/MY_Module/protein_information/Total_electrostatic_energy/output\n")
        f.write("    end\n")
        f.write("    swin 0.3\n")


        f.write("    ion 1 charge 1 radius 2.0\n")
        f.write("    ion 2 charge -1 radius 2.0\n")


        f.write("    calcforce  no\n")
        f.write("    calcenergy comps\n")
        f.write("end\n")

        f.write("    print elecEnergy com end\n")


    run_apbs(input_file)

    total_energy = extract_total_energy_from_output("/home/ntu/Documents/feiailu/MY_Module/protein_information/Total_electrostatic_energy/temp_output.txt")


    return total_energy

