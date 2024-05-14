# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 15:34:20 2024

@author: Naz
"""

import os
import shutil
import re
import numpy as np
from scipy.spatial import distance
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation
import importlib.util

# rot_mod_dir = importlib.util.spec_from_file_location("orient", "E:\3_QM\Reactionsearch_QM\scripts\orient_molecule\orient-molecule\orient.py")
# orient = importlib.util.module_from_spec(rot_mod_dir)

     
        
# file1 = "e416.mol2"
# atom1 = 26
# file2 = "po_ant.xyz"
# atom2 = 33


class MoleculeAndAtom:
    def __init__(self, file, atom):
        self.file = file
        # self.atom = atom
        self.atom_idx = atom
        self.atom_labels = []
        self.mol_coord_list = []        
        with open(file,"r") as file:
            item = 0
            for idx, line in enumerate(file):
                if re.match(r".*([A-Z])\s+([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)",line):            
                    atom_coord = re.search(r".*([A-Z])\s+([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)",line)
                    atom_name = str(item+1)+atom_coord.group(1)
                    self.atom_labels.append(atom_name)
                    atom = [float(atom_coord.group(2)),float(atom_coord.group(3)),float(atom_coord.group(4))]
                    self.mol_coord_list.append(atom)
                    # if atom_name == atom:
                    #     atom_idx = item   
                    item = item +1                
        self.coord_matrix = np.array(self.mol_coord_list)  
        self.main_atom = self.coord_matrix[self.atom_idx]
        self.dist_matrix = distance.cdist(self.coord_matrix,self.coord_matrix)
        self.findTwoNeighbors() 
        
        
        # self.angleFromThreePoints()  
        # tree = KDTree(self.coord_matrix)
        # neighbors_full = tree.query_ball_point(self.main_atom, 1.5)
        # self.neighbors = neighbors_full.remove(self.atom_idx)
        
    def findTwoNeighbors(self):
        distance_list = self.dist_matrix[self.atom_idx].tolist()
        first_atom_dist = float('inf')
        second_atom_dist = float('inf')
        first_atom_idx = 0
        second_atom_idx = 0
        for dist in distance_list:
            if first_atom_dist >= dist and dist >= 0.1 :
                second_atom_dist = first_atom_dist
                first_atom_dist = dist
                second_atom_idx = first_atom_idx
                first_atom_idx = distance_list.index(dist)            
            elif second_atom_dist > dist and dist>=0.1:
                second_atom_dist = dist
                second_atom_idx = distance_list.index(dist)
        self.neighbor1 = self.coord_matrix[first_atom_idx]
        self.neighbor2 = self.coord_matrix[second_atom_idx]
        return [first_atom_idx,second_atom_idx]    
    
    def angleFromThreePoints(self):
        A=self.main_atom
        B=self.neighbor1
        C=self.neighbor2
        AB = B-A
        AC = C-A        
        angle_rad = np.arccos(np.dot(AB,AC)/(np.linalg.norm(AB)*np.linalg.norm(AC)))
        angle = np.degrees(angle_rad)
        self.angle = angle
        return angle
    
    
class MoleculeOrganisation():
    def __init__(self, mol1, mol2, dist):
        self.mol1 = mol1
        self.mol2 = mol2
        self.dist = dist
        mol1_coord = mol1.coord_matrix
        mol2_coord = mol2.coord_matrix
        
        # GP, norm_vect, rot_axis = self.determineGhostPoint(angle=-20)
        # mol2_allign = self.alignMol2(mol2_coord,GP,norm_vect)
        # mol2_rot = self.rotateAroundAxis(mol2_allign, GP, norm_vect, 0)
        # self.addCoordToFile(mol2_rot)
        opt_mol2 = self.optimalOrientation(mol1_coord, mol2_coord)
        self.createGausFile(opt_mol2)
     

        
    def determineGhostPoint(self,angle):
        dist= self.dist
        A = self.mol1.main_atom  
        B = self.mol1.neighbor1  
        C = self.mol1.neighbor2                     
        AB = B-A
        D = C + AB
        AD = D - A
        ad_norm = AD/np.linalg.norm(AD)
        AD_dist = ad_norm*dist
        Q = A - AD_dist
        AQ = Q - A
        aq_norm = AQ/np.linalg.norm(AQ)
        
        AB = B - A
        AC = C - A  
        plane = np.cross(AB,AC)
        a,b,c = plane
        d = np.dot(plane,A)    
        Al = D.tolist()        
        n = Al[0]*a+Al[1]*b+Al[2]*c
        # if (abs(d-n)<0.00001):
            # print("ghost point lies in the plane")      
        rot_rad = np.radians(angle)
        rotation_vector = rot_rad*plane
        rotation = Rotation.from_rotvec(rotation_vector)
        AO = rotation.apply(AQ)
        O = AO + A
        normal = AO/np.linalg.norm(AO)
        
        return O, normal, plane
        
       
      
        
    def alignMol2(self,input_mol2,Q,norm_vect):              
        H = np.array(self.mol2.main_atom)
        # mol2_coord = self.mol2.coord_matrix
        diff = np.array(Q)-H
        mol2_inpoint = input_mol2 + diff  
        n1_idx = self.mol2.findTwoNeighbors()[0]
        
        N1 = mol2_inpoint[n1_idx]        
        A = self.mol1.main_atom
                
        QN1 = N1 - Q
        qn1_dist = np.linalg.norm(QN1)
                    
        AO = norm_vect*(qn1_dist+self.dist)
        O = AO + A                  # target point
        QO = O - Q

        rotation_axis = np.cross(QN1, QO)
        rotation_angle = np.arccos(np.dot(QN1, QO) / (np.linalg.norm(QN1) * np.linalg.norm(QO)))
        full_angle = np.rad2deg(rotation_angle)

        rotation = Rotation.from_rotvec(full_angle * rotation_axis,degrees=True)
        M_rotated = rotation.apply(mol2_inpoint-Q) + Q
       
        M1 = M_rotated[n1_idx]
       
        return M_rotated
        
    
    
   
    def rotateAroundAxis(self,input_matrix,GP,rot_axis,angle_deg):
        rotation = Rotation.from_rotvec(angle_deg*rot_axis,degrees = True)
        rotated_matrix = rotation.apply(input_matrix - GP) + GP
        return rotated_matrix
     
    
  
    def optimalOrientation(self,mol1_coord,mol2_coord):
        dist = self.dist     
        opt_dist = 0
        opt_sum_dist = 0   
        # opt_angle = 0
        opt_mol2 = mol2_coord
        for g in range(-15,15):         
            angle1 = 2*g        
            GP, norm, rot_axis = self.determineGhostPoint(angle1)
            mol2_allign = self.alignMol2(mol2_coord,GP,norm)
            
            for ang in range(0,37): 
                distances = []
                angle2 = ang*10
                rotation = Rotation.from_rotvec(angle2*norm,degrees = True)
                mol2_rot = rotation.apply(mol2_allign - GP) + GP
                all_dist = distance.cdist(mol1_coord, mol2_rot)
                for i in range(0,len(mol1_coord)):
                    for j in range(0,len(mol2_rot)):
                       if all_dist[i][j] != dist: 
                           distances.append(all_dist[i][j])

                sort_dist = sorted(distances)
                sum_low_dist = 0 
                for k in  range(0,len(sort_dist)):
                    if (k < len(sort_dist)/3):
                        sum_low_dist += sort_dist[k]
                        
                if (abs(opt_dist-sort_dist[0]) < 0.01):
                    if (sum_low_dist > opt_sum_dist):
                        opt_dist = sort_dist[0]
                        opt_sum_dist = sum_low_dist
                        opt_mol2 = mol2_rot
                
                if (opt_dist < sort_dist[0]):
                    opt_dist = sort_dist[0]
                    opt_sum_dist = sum_low_dist
                    opt_mol2 = mol2_rot
        
        if opt_dist < 2.2:    
            print(opt_dist, opt_sum_dist)        
        return opt_mol2
               
        
        
    
    def addCoordToFile(self,input_coord):
        file = self.mol1.file
        filename1 = os.path.splitext(os.path.basename(self.mol1.file))[0]
        filename2 = os.path.splitext(os.path.basename(self.mol2.file))[0]
        org_file = filename1+"_"+filename2+"_org.xyz"   
        # input_coord = self.trans_coord
        input_labels = self.mol2.atom_labels
        atoms_number = str(len(self.mol1.atom_labels)+len(self.mol2.atom_labels))
        lines = []
        if (input_coord.shape[0] == len(input_labels)):
            with open(file,"r") as mol1_file:
                for line in mol1_file:
                    if re.match(r".*([A-Z])\s+([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)",line):
                        coord1 = re.search(r".*([A-Z])\s+([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)",line)
                        lines.append("{}         {}        {}       {}\n".format(coord1.group(1),coord1.group(2),coord1.group(3),coord1.group(4)))
            for i in range(len(input_labels)):
              coord = input_coord[i].tolist()
              label = re.search(r"[0-9]+([A-Za-z]+)", input_labels[i])
              lines.append("{}         {}        {}       {}\n".format(label.group(1), str(round(coord[0],5)),str(round(coord[1],5)),str(round(coord[2],5))))
            with open(org_file,"w") as file:
               file.write(atoms_number+"\n"+"\n")
               file.writelines(line for line in lines)  
      
   
    
    def createGausFile(self,input_coord):
        file = self.mol1.file
        filename1 = os.path.splitext(os.path.basename(self.mol1.file))[0]
        filename2 = os.path.splitext(os.path.basename(self.mol2.file))[0]
        gaus_name = filename1+"_"+filename2   
        gaus_title = filename1+"_"+filename2+".gjf"   
        bash_title = filename1+"_"+filename2+".sh"   
        # input_coord = self.trans_coord
        input_labels = self.mol2.atom_labels
        atoms_number = str(len(self.mol1.atom_labels)+len(self.mol2.atom_labels))
        route_lines = []
        bash_lines = []
        lines = []
        if (input_coord.shape[0] == len(input_labels)):
            with open("gaus.gjf","r") as gfile:
                for line in gfile:
                    if "name" in line:
                        line = line.replace("name", gaus_name)
                        # print(line)
                    route_lines.append(line)    
            with open(file,"r") as mol1_file:
                for line in mol1_file:
                    if re.match(r".*([A-Z])\s+([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)",line):
                        coord1 = re.search(r".*([A-Z])\s+([0-9.-]+)\s+([0-9.-]+)\s+([0-9.-]+)",line)
                        lines.append("{}         {}        {}       {}\n".format(coord1.group(1),coord1.group(2),coord1.group(3),coord1.group(4)))
            for i in range(len(input_labels)):
                coord = input_coord[i].tolist()
                label = re.search(r"[0-9]+([A-Za-z]+)", input_labels[i])
                lines.append("{}         {}        {}       {}\n".format(label.group(1), str(round(coord[0],5)),str(round(coord[1],5)),str(round(coord[2],5))))
            with open(gaus_title,"w") as gaus_file:
                gaus_file.writelines(line for line in route_lines)
                gaus_file.writelines(line for line in lines)
                gaus_file.write(" \n")
            with open("rad.sh","r") as b_file:
                for line in b_file:
                    if "joname" in line:
                        line = line.replace("joname", gaus_name)
                        # print(line)
                    bash_lines.append(line)                    
            with open(bash_title,"w",newline='\n') as bash_file:
                bash_file.writelines(bash_lines)
                
        
if __name__ == "__main__":
    
    path = os.path.join(os.getcwd(),"e_po")
    os.chdir(path)
    
    for file in os.scandir(path):
    
        if re.match(r"e([0-9]+)\.",file.name):
            file1 = file.name
            print(file1)
            with open(file1,"r") as e_file:
                atom1 = int(e_file.readline())
            file2 = "po.xyz"
            atom2 = 5
            molecule1 = MoleculeAndAtom(file1,atom1)
            molecule2 = MoleculeAndAtom(file2,atom2)
            organizator = MoleculeOrganisation(molecule1, molecule2, 1.7)



