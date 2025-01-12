import torch

# Load the dataset and variables from .pt file
data = torch.load("clintox_dataset.pt")

# Extract variables
smiles_list = data["smiles_list"]
molecules = data["molecules"]
labels = data["labels"]

print(f"Loaded {len(molecules)} molecules with labels.")

from rdkit import Chem
from rdkit.Chem import rdmolops
import networkx as nx
import matplotlib.pyplot as plt
from rdkit.Chem import Draw

def molecule_to_graph(mol):
    """Convert an RDKit molecule to a NetworkX graph."""
    graph = nx.Graph()
    for atom in mol.GetAtoms():
        graph.add_node(atom.GetIdx(), atom=atom)
    for bond in mol.GetBonds():
        graph.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), bond=bond)
    return graph

def extract_rings_with_sssr(mol):
    """Extract rings using RDKit's GetSymmSSSR."""
    sssr = list(rdmolops.GetSymmSSSR(mol))  # Returns tuples of atom indices for each ring
    return [set(ring) for ring in sssr]

def add_singletons(graph, rings):
    """Add singleton atoms as cliques if they are not part of any ring."""
    covered_atoms = set(atom for ring in rings for atom in ring)
    all_atoms = set(graph.nodes)
    singletons = [{atom} for atom in all_atoms - covered_atoms]
    return rings + singletons

def visualize_molecule(mol):
    """Visualize an RDKit molecule."""
    return Draw.MolToImage(mol, size=(300, 300))

def build_junction_tree(cliques, mol):
    """Build a junction tree from the cliques, ensuring edges for adjacent rings."""
    jt = nx.Graph()

    for idx, clique in enumerate(cliques):
        if not clique or max(clique) >= mol.GetNumAtoms() or min(clique) < 0:
            print(f"Skipping invalid clique at index {idx}: {clique}")
            continue
        jt.add_node(idx, atoms=clique)

    for i in range(len(cliques)):
        for j in range(i + 1, len(cliques)):
            overlap = set(cliques[i]).intersection(cliques[j])
            if overlap:
                jt.add_edge(i, j, weight=len(overlap))
            elif are_rings_adjacent(mol, cliques[i], cliques[j]):
                jt.add_edge(i, j, weight=1)  # Minimal weight for adjacency

    if jt.edges:
        jt = nx.maximum_spanning_tree(jt, weight='weight')

    return jt

def are_rings_adjacent(mol, ring1, ring2):
    """Check if two rings are adjacent by verifying bonds between their atoms."""
    for atom1 in ring1:
        for atom2 in ring2:
            bond = mol.GetBondBetweenAtoms(atom1, atom2)
            if bond is not None:
                return True  # Rings are adjacent
    return False

def visualize_junction_tree(mol, mol_graph, jt):
    """Visualize the junction tree of a molecule."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Visualize molecule
    img = visualize_molecule(mol)
    axes[0].imshow(img)
    axes[0].axis('off')
    axes[0].set_title('Molecule')

    # Prepare labels for cliques
    labels = {}
    for node, data in jt.nodes(data=True):
        atom_indices = data['atoms']
        atom_symbols = [mol.GetAtomWithIdx(idx).GetSymbol() for idx in atom_indices]
        labels[node] = ''.join(atom_symbols)

    # Visualize junction tree
    pos = nx.spring_layout(jt)
    nx.draw(jt, pos, ax=axes[1], with_labels=True, labels=labels, node_color='lightblue', edge_color='gray')
    axes[1].set_title('Junction Tree')

    plt.show()

def junction_tree_to_gspan(jt, mol, output_file):
    """Convert a junction tree into gSpan input format with SMILES node labels and bond type edge labels."""
    with open(output_file, "w") as f:
        graph_id = 0  # Incremental graph ID
        f.write(f"t # {graph_id}\n")  # Start a new graph

        # Write nodes with SMILES labels
        for node_id, data in jt.nodes(data=True):
            atoms = data['atoms']
            smiles = get_smiles_for_substructure(mol, atoms)  # SMILES string as node label
            f.write(f"v {node_id} {smiles}\n")
        
        # Write edges with bond type labels
        for node1, node2 in jt.edges():
            bond_type = get_edge_bond_type(mol, jt.nodes[node1]['atoms'], jt.nodes[node2]['atoms'])
            f.write(f"e {node1} {node2} {bond_type}\n")
            
from rdkit.Chem import rdmolops

def are_atoms_connected(mol, atom_indices):
    """Check if a set of atom indices form a connected fragment."""
    subgraph = mol.GetSubstructMatches(Chem.MolFromSmiles(Chem.MolFragmentToSmiles(mol, atomsToUse=list(atom_indices))))
    return any(set(match) == atom_indices for match in subgraph)


def get_smiles_for_substructure(mol, atom_indices):
    """Generate a SMILES string for a substructure given atom indices."""
    try:
        # Debugging: Log input atom indices
        print(f"Processing atom indices: {atom_indices}")

        # # Handle empty atom indices
        # if not atom_indices:
        #     print("Error: Atom indices are empty.")
        #     return "EMPTY"

        # # Validate atom indices range
        # max_index = mol.GetNumAtoms() - 1
        # if any(idx < 0 or idx > max_index for idx in atom_indices):
        #     print(f"Error: Atom indices out of range. Indices: {atom_indices}, Max allowed: {max_index}")
        #     return "OUT_OF_RANGE"

        # if not are_atoms_connected(mol, atom_indices):
        #     print(f"Atom indices are not connected: {atom_indices}")
        #     return "DC"

        # Debugging: Log before calling PathToSubmol
        print(f"Calling PathToSubmol with indices: {atom_indices}")
# 
        # radius = 15
        # atom_idx = 0
        # Generate substructure and SMILES
        try:
            # env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
            # amap = {}
            # submol=Chem.PathToSubmol(mol,env,atomMap=amap)
            # 
            submol = Chem.PathToSubmol(mol, list(atom_indices))
            
        except Exception as e:
            # Log the issue
            with open("segfault_molecules_log.txt", "a") as log_file:
                log_file.write(f"Molecule: {Chem.MolToSmiles(mol)}, Atom Indices: {atom_indices}, Error: {e}\n")
            # Save the visualization
            Draw.MolToFile(mol, f"problematic_molecule_{i + 1}.png")
        
        smiles = Chem.MolToSmiles(submol)

        # Debugging: Log successful SMILES generation
        print(f"Generated SMILES: {smiles}")
        return smiles

    except Exception as e:
        # Log detailed error information
        print(f"Error in get_smiles_for_substructure: {e}, Indices: {atom_indices}")
        return "ERROR"

def get_edge_bond_type(mol, atoms1, atoms2):
    """Get the bond type connecting two sets of atoms."""
    for atom1 in atoms1:
        for atom2 in atoms2:
            bond = mol.GetBondBetweenAtoms(atom1, atom2)
            if bond is not None:
                return bond.GetBondType().name  # Return bond type (e.g., SINGLE, DOUBLE)
    return "NONE"  # No bond found

def get_smiles_with_radius(mol, atom_indices, max_radius=5):
    """Generate SMILES strings for atom neighborhoods, ensuring isolation within the substructure."""
    fallback_smiles = []
    processed_atoms = set()
    try:
        for atom_idx in atom_indices:
            if atom_idx in processed_atoms:
                continue  # Skip already processed atoms

            for radius in range(1, max_radius + 1):
                env = Chem.FindAtomEnvironmentOfRadiusN(mol, radius, atom_idx)
                if not env:
                    continue

                # Extract all atoms in the environment
                extracted_atoms = {mol.GetBondWithIdx(b).GetBeginAtomIdx() for b in env}.union(
                    {mol.GetBondWithIdx(b).GetEndAtomIdx() for b in env}
                )

                # Validate extracted atoms are within the problematic substructure
                if not extracted_atoms.issubset(set(atom_indices)):
                    print(f"Warning: Extracted atoms ({extracted_atoms}) overlap with other substructures.")
                    continue

                # Update processed atoms
                processed_atoms.update(extracted_atoms)

                # Generate submolecule and SMILES
                submol = Chem.PathToSubmol(mol, env)
                smiles = Chem.MolToSmiles(submol)
                fallback_smiles.append(smiles)
    except Exception as e:
        print(f"Fallback failed for atom indices {atom_indices}: {e}")
        return ["FALLBACK_ERROR"]

    return fallback_smiles

import os, gc

import tensorflow as tf

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# tf.config.threading.set_inter_op_parallelism_threads(1)
# tf.config.threading.set_intra_op_parallelism_threads(1)

# Ensure output directories exist
output_images_dir = "junction_trees_images1_2"
os.makedirs(output_images_dir, exist_ok=True)

# File to store all junction trees
junction_trees_file = "all_junction_trees_gspan1_2.txt"

offset = 1083 # 804, 870, 871, 1039, 1083 skipped
             # 711 valid
             # 764 (DC), 956 (ERROR), 979 (DC), 1057 (ERROR and DC) fixed but with issues

# Open the gSpan file for writing all junction trees
with open(junction_trees_file, "a") as gspan_file:
    for i, mol in enumerate(molecules[offset:offset + 1]):  # Process molecules
        try:
            print(f"Processing molecule {i + offset}")

            # Check if the molecule is valid
            if mol is None or mol.GetNumAtoms() == 0:
                print(f"Invalid molecule at index {i + offset}")
                print(f"{i + offset}: Invalid molecule\n")
                continue
            
            # Step 1: Convert molecule to graph
            mol_graph = molecule_to_graph(mol)

            # Step 2: Extract rings using GetSymmSSSR
            try:
                rings = extract_rings_with_sssr(mol)
            except Exception as e:
                print(f"Error extracting rings for molecule {i + offset}: {e}")
                print(f"{i + offset}: Error extracting rings - {Chem.MolToSmiles(mol)}\n")
                continue

            # Step 3: Add singleton atoms
            try:
                all_cliques = add_singletons(mol_graph, rings)
            except Exception as e:
                print(f"Error adding singletons for molecule {i + offset}: {e}")
                print(f"{i + offset}: Error adding singletons - {Chem.MolToSmiles(mol)}\n")
                continue

            # Step 4: Build junction tree
            try:
                junction_tree = build_junction_tree(all_cliques, mol)
            except Exception as e:
                print(f"Error building junction tree for molecule {i + offset}: {e}")
                print(f"{i + offset}: Error building junction tree - {Chem.MolToSmiles(mol)}\n")
                continue

             # Step 5: Save junction tree in gSpan format
            graph_id = i + offset
            gspan_file.write(f"t # {graph_id}\n")  # Start a new graph

            log_file = "debugging_errors.txt"
            
            for node_id, data in junction_tree.nodes(data=True):
                atom_indices = data.get('atoms', [])
                # smiles = get_smiles_for_substructure(mol, atom_indices)
                smiles_list = get_smiles_with_radius(mol, atom_indices)
                
                if "ERROR" in smiles_list or not smiles_list:
                    print(f"Skipping problematic node with atom indices: {atom_indices}")
                    with open("problematic_nodes_log.txt", "a") as log:
                        log.write(f"Molecule {i + 1}, Node {node_id}: Error or no substructures, Atom indices: {atom_indices}\n")
                    continue

                # print(f"Node {node_id}, SMILES: {smiles_list}")
            
            with open(log_file, "a") as log:
                for node_id, data in junction_tree.nodes(data=True):
                    try:
                        atom_indices = data.get('atoms', [])
                        print(f"Node {node_id}, Atom Indices: {atom_indices}")

                        smiles = get_smiles_for_substructure(mol, atom_indices)

                        # if smiles in ["EMPTY", "OUT_OF_RANGE", "DC", "ERROR"]:
                        #     print(f"Skipping problematic node with atom indices: {atom_indices}")
                        #     with open("problematic_nodes_log.txt", "a") as log:
                        #         log.write(f"Molecule {i + 1}, Node {node_id}: {smiles}, Atom indices: {atom_indices}\n")
                        #     continue

                        print(f"Node {node_id}, SMILES: {smiles}")
                        gspan_file.write(f"v {node_id} {smiles}\n")
                        
                    except Exception as e:
                        # Log unexpected errors at this level
                        print(f"Unexpected error in node {node_id}: {e}")
                        log.write(f"Graph {graph_id}, Node {node_id}: Unexpected error - {e}\n")
                        continue
                
                for node1, node2, edge_data in junction_tree.edges(data=True):
                    # Edge label is the bond type
                    bond_type = get_edge_bond_type(mol, junction_tree.nodes[node1]['atoms'], junction_tree.nodes[node2]['atoms'])
                    gspan_file.write(f"e {node1} {node2} {bond_type}\n")

        except Exception as e:
            print(f"Error processing node {node_id}: {e}")

        try:
            # Debugging: Print number of nodes in the junction tree
            print(f"Processing junction tree with {len(junction_tree.nodes())} nodes")

            for node_id, data in junction_tree.nodes(data=True):
                print(f"Node {node_id}: {data}")
                atom_indices = data.get('atoms', [])

                # Debugging: Check validity of atom indices
                if not atom_indices:
                    print(f"Node {node_id}: Empty atom indices")
                    continue

                print(f"Node {node_id}: Atom indices are valid: {atom_indices}")
        except Exception as e:
            print(f"Error during node processing: {e}")
            
        except Exception as e:
                print(f"Unexpected error processing molecule {i + offset}: {e}")
                try:
                    # Log the SMILES of the molecule if possible
                    smiles = Chem.MolToSmiles(mol) if mol else "None"
                except:
                    smiles = "Invalid SMILES"
                print(f"{i + 1}: Unexpected error - {smiles}\n")

        # Step 6: Visualize and save molecule and junction tree images
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        # Molecule visualization
        img = visualize_molecule(mol)
        axes[0].imshow(img)
        axes[0].axis('off')
        axes[0].set_title('Molecule')

        # Junction tree visualization
        pos = nx.spring_layout(junction_tree)
        labels = {
            node: get_smiles_for_substructure(mol, junction_tree.nodes[node]['atoms'])
            for node in junction_tree.nodes
        }
        nx.draw(
            junction_tree, pos, ax=axes[1], with_labels=True, labels=labels,
            node_color='lightblue', edge_color='gray'
        )
        axes[1].set_title('Junction Tree')

        # Save image
        image_file = os.path.join(output_images_dir, f"junction_tree_{graph_id}.png")
        plt.savefig(image_file)
        plt.close()
        
        gc.collect()
