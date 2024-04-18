""" The ``retrekpy.gln`` package ``gln_utilities`` module. """

import csv
import multiprocessing
import numpy
import os
import random
import re

from tqdm import tqdm

from rdchiral.template_extractor import extract_from_reaction

from rdkit.Chem import MolFromSmiles, MolToSmiles


class GLNUtilities:
    """ The 'GLNUtilities' class. """

    @staticmethod
    def get_rxn_smiles(prod, reactants):
        """ The 'get_rxn_smiles' function. """

        prod_smi = MolToSmiles(prod, True)

        # Get rid of reactants when they don't contribute to this prod
        prod_maps = set(re.findall('\:([[0-9]+)\]', prod_smi))
        reactants_smi_list = []

        for mol in reactants:
            if mol is None:
                continue

            used = False

            for a in mol.GetAtoms():
                if a.HasProp('molAtomMapNumber'):
                    if a.GetProp('molAtomMapNumber') in prod_maps:
                        used = True

                    else:
                        a.ClearProp('molAtomMapNumber')
            if used:
                reactants_smi_list.append(MolToSmiles(mol, True))

        reactants_smi = '.'.join(reactants_smi_list)

        return '{}>>{}'.format(reactants_smi, prod_smi)

    @staticmethod
    def get_writer(fname, header):
        """ The 'get_writer' function. """

        fout = open(fname, 'w')
        writer = csv.writer(fout)
        writer.writerow(header)

        return fout, writer

    @staticmethod
    def get_tpl(task):
        """ The 'get_tpl' function. """

        idx, row_idx, rxn_smiles = task
        react, reagent, prod = rxn_smiles.split('>')
        reaction = {'_id': row_idx, 'reactants': react, 'products': prod}

        try:
            template = extract_from_reaction(reaction)

        except:
            return idx, {'err_msg': "exception"}

        return idx, template

    @staticmethod
    def gln_clean_uspto(uspto_rsmi_file_path, save_folder_path):
        """ The original 'gln_clean_uspto' function. """

        uspto_version = "grants" if "1976" in uspto_rsmi_file_path else "applications"
        fname = uspto_rsmi_file_path

        seed = 19260817
        numpy.random.seed(seed)
        random.seed(seed)

        split_mode = 'multi'  # single or multi

        pt = re.compile(r':(\d+)]')
        # cnt = 0
        clean_list = []
        set_rxn = set()
        num_single = 0
        num_multi = 0
        bad_mapping = 0
        bad_prod = 0
        missing_map = 0
        raw_num = 0

        with open(fname) as f:
            reader = csv.reader(f, delimiter='\t')
            header = next(reader)
            pbar = tqdm(reader)
            # bad_rxn = 0
            for row in pbar:
                rxn_smiles = row[header.index('ReactionSmiles')]
                all_reactants, reagents, prods = rxn_smiles.split('>')
                all_reactants = all_reactants.split()[0]  # remove ' |f:1...'
                prods = prods.split()[0]  # remove ' |f:1...'
                if '.' in prods:
                    num_multi += 1
                else:
                    num_single += 1
                if split_mode == 'single' and '.' in prods:  # multiple prods
                    continue
                rids = ','.join(sorted(re.findall(pt, all_reactants)))
                pids = ','.join(sorted(re.findall(pt, prods)))
                if rids != pids:  # mapping is not 1:1
                    bad_mapping += 1
                    continue
                reactants = [MolFromSmiles(smi) for smi in all_reactants.split('.')]

                for sub_prod in prods.split('.'):
                    mol_prod = MolFromSmiles(sub_prod)
                    if mol_prod is None:  # rdkit is not able to parse the product
                        bad_prod += 1
                        continue
                    # Make sure all have atom mapping
                    if not all([a.HasProp('molAtomMapNumber') for a in mol_prod.GetAtoms()]):
                        missing_map += 1
                        continue

                    raw_num += 1
                    rxn_smiles = GLNUtilities.get_rxn_smiles(mol_prod, reactants)
                    if rxn_smiles not in set_rxn:
                        clean_list.append((
                            row[header.index('PatentNumber')],
                            rxn_smiles,
                            row[header.index('ReactionSmiles')],
                            row[header.index('ParagraphNum')],
                            row[header.index('Year')]
                        ))
                        set_rxn.add(rxn_smiles)
                pbar.set_description('select: %d, dup: %d' % (len(clean_list), raw_num))

        print('# clean', len(clean_list))
        print('single', num_single, 'multi', num_multi)
        print('bad mapping', bad_mapping)
        print('bad prod', bad_prod)
        print('missing map', missing_map)
        print('raw extracted', raw_num)

        random.shuffle(clean_list)

        num_val = num_test = int(len(clean_list) * 0.1)

        for phase in ['val', 'test', 'train']:
            fout = os.path.join(save_folder_path,  f"uspto_{uspto_version}" + "_raw_%s.csv" % phase)

            with open(fout, 'w') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'document_id',
                    'paragraph_num',
                    'publication_year',
                    'original_reaction_smiles',
                    'reactants>reagents>production'
                ])

                if phase == 'val':
                    r = range(num_val)
                elif phase == 'test':
                    r = range(num_val, num_val + num_test)
                else:
                    r = range(num_val + num_test, len(clean_list))
                for i in r:
                    rxn_smiles = clean_list[i][1].split('>')
                    result = []
                    for r in rxn_smiles:
                        if len(r.strip()):
                            r = r.split()[0]
                        result.append(r)
                    rxn_smiles = '>'.join(result)

                    writer.writerow([
                        clean_list[i][0],
                        clean_list[i][3],
                        clean_list[i][4],
                        clean_list[i][2],
                        rxn_smiles
                    ])

    @staticmethod
    def gln_build_raw_template(uspto_rsmi_file_path, save_folder_path, num_cpu_cores):
        """ The original 'gln_build_raw_template' function. """

        uspto_version = "grants" if "1976" in uspto_rsmi_file_path else "applications"

        seed = 19260817
        numpy.random.seed(seed)
        random.seed(seed)

        for dataset_split in ["train", "test", "val"]:
            fname = os.path.join(save_folder_path, f"uspto_{uspto_version}_raw_{dataset_split}.csv")

            with open(fname) as f:
                reader = csv.reader(f)
                next(reader)
                rows = [row for row in reader]

            pool = multiprocessing.Pool(num_cpu_cores)
            tasks = []
            for idx, row in tqdm(enumerate(rows)):
                # One of the indices always won't finish, thus exclude using the index from tqdm.
                if idx != 88303:
                    row_idx, row_par_num, row_year, row_og_smi, rxn_smiles = row
                    tasks.append((idx, row_idx, rxn_smiles))

            fout, writer = GLNUtilities.get_writer(
                os.path.join(save_folder_path, f"uspto_{uspto_version}_single_product_{dataset_split}.csv"),
                ['document_id', 'paragraph_num', 'publication_year', 'original_reaction_smiles', 'new_reaction_smiles',
                 'retro_templates']
            )

            fout_failed, failed_writer = GLNUtilities.get_writer(
                os.path.join(save_folder_path, f"uspto_{uspto_version}_failed_template_{dataset_split}.csv"),
                ['document_id', 'paragraph_num', 'publication_year', 'original_reaction_smiles', 'rxn_smiles', 'err_msg']
            )

            for result in tqdm(pool.imap_unordered(GLNUtilities.get_tpl, tasks), total=len(tasks)):
                idx, template = result

                row_idx, row_par_num, row_year, row_og_smi, rxn_smiles = rows[idx]

                if 'reaction_smarts' in template:
                    writer.writerow([row_idx, row_par_num, row_year, row_og_smi, rxn_smiles, template['reaction_smarts']])
                    fout.flush()

                    # print(f"finished {idx}")
                else:
                    if 'err_msg' in template:
                        failed_writer.writerow(
                            [row_idx, row_par_num, row_year, row_og_smi, rxn_smiles, template['err_msg']]
                        )
                    else:
                        failed_writer.writerow([row_idx, row_par_num, row_year, row_og_smi, rxn_smiles, "NaN"])

                    fout_failed.flush()

                    # print(f"finished {idx}")

            fout.close()
            fout_failed.close()
