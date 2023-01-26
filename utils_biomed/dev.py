import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import spacy
import scispacy
from scispacy.linking import EntityLinker

repo_root = '..'


def load_entity_linker(threshold=0.90):
    nlp = spacy.load("en_core_sci_sm")
    linker = EntityLinker(
        resolve_abbreviations=True,
        name="umls",
        threshold=threshold)
    nlp.add_pipe(linker)
    return nlp, linker
nlp, linker = load_entity_linker()


def entity_linking_to_umls(sentence, nlp, linker):
    doc = nlp(sentence)
    entities = doc.ents
    all_entities_results = []
    for mm in range(len(entities)):
        entity_text = entities[mm].text
        entity_start = entities[mm].start
        entity_end = entities[mm].end
        all_linked_entities = entities[mm]._.kb_ents
        all_entity_results = []
        for ii in range(len(all_linked_entities)):
            curr_concept_id = all_linked_entities[ii][0]
            curr_score = all_linked_entities[ii][1]
            curr_scispacy_entity = linker.kb.cui_to_entity[all_linked_entities[ii][0]]
            curr_canonical_name = curr_scispacy_entity.canonical_name
            curr_TUIs = curr_scispacy_entity.types
            curr_entity_result = {"Canonical Name": curr_canonical_name, "Concept ID": curr_concept_id,
                                  "TUIs": curr_TUIs, "Score": curr_score}
            all_entity_results.append(curr_entity_result)
        curr_entities_result = {"text": entity_text, "start": entity_start, "end": entity_end,
                                "start_char": entities[mm].start_char, "end_char": entities[mm].end_char,
                                "linking_results": all_entity_results}
        all_entities_results.append(curr_entities_result)
    return all_entities_results


umls_to_ddb = {}
with open(f'{repo_root}/data/ddb/ddb_to_umls_cui.txt') as f:
    for line in f.readlines()[1:]:
        elms = line.split("\t")
        umls_to_ddb[elms[2]] = elms[1]
def map_to_ddb(ent_obj):
    res = []
    for ent_cand in ent_obj['linking_results']:
        CUI  = ent_cand['Concept ID']
        name = ent_cand['Canonical Name']
        if CUI in umls_to_ddb:
            ddb_cid = umls_to_ddb[CUI]
            res.append((ddb_cid, name))
    return res

sent = "A 5-year-old girl is brought to the emergency department by her mother because of multiple episodes of nausea and vomiting that last about 2 hours. During this period, she has had 6–8 episodes of bilious vomiting and abdominal pain. The vomiting was preceded by fatigue."
ent_link_results = entity_linking_to_umls(sent, nlp, linker)
for obj in ent_link_results:
    # 这里非空就表示一个 (id, name) pair,
    print(map_to_ddb(obj))