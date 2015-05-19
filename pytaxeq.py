#!/usr/bin/env python2

import re
import numpy as np
from collections import Counter
from jinja2 import Environment, FileSystemLoader

header = re.compile(r'^(?P<ntax>\d+)\s+(?P<nchar>\d+)\s+(?P<description>.+?)\s*$')
taxon = re.compile(r'^(?P<taxon>[\S]+)\s+(?P<state>(?:[\d?]\s*)+?)\s*$')

env = Environment(loader=FileSystemLoader('templates'))

class PyTaxEq(object):
    """
    Converts a specially formatted file specifying a phylogenetic matrix into a
    numpy-based array from which taxonomic equivalence can be calculated.
    """
    def __init__(self, in_file=''):
        ntax = 0
        self._ordered = set()
        self._uninformative = set()
        self._taxa = []
        if not in_file:
            return
        with open(in_file, 'rb') as data:
            input_data = data.readlines()
            tax_count = 0

            for num, line in enumerate(input_data):
                if not ntax:
                    match = header.match(line)
                    if match:
                        ntax = int(match.group('ntax'))
                        nchar = int(match.group('nchar'))
                        self._description = match.group('description')
                        self._matrix = np.zeros((ntax, nchar),
                                                dtype=np.float32)
                else:
                    if taxon.match(line):
                        if tax_count >= ntax:
                            break
                        new_taxon = taxon.match(line).groupdict()
                        self._taxa.append(new_taxon['taxon'])
                        states = re.findall(r'[\d?]', new_taxon['state'])

                        if len(states) != nchar:
                            raise ValueError("Taxon {0[taxon]} has {1}" \
                                " characters, expected {2[nchar]}".format(
                                    new_taxon, len(states), nchar))

                        for char_num, state in enumerate(states):
                            if state == '?':
                                self._matrix[tax_count, char_num] = np.nan
                            else:
                                self._matrix[tax_count, char_num] = int(state)

                        tax_count += 1
                    elif line.strip() == 'ordered':
                        line = input_data[num + 1]
                        for ord_tax in re.findall(r'(\d+)\s+', line):
                            self._ordered.add(int(ord_tax))
                        break

            if tax_count != ntax:
                raise ValueError('Found {0} taxa, expected {1}'.format(
                    tax_count, ntax))

    @property
    def matrix(self):
        """Return a copy of the numpy matrix array."""
        return self._matrix

    @matrix.setter
    def matrix(self, matrix):
        """Set a copy of the numpy matrix array."""
        self._matrix = matrix

    @property
    def ntax(self):
        """Return the number of taxa in the matrix."""
        return self._matrix.shape[0]

    @property
    def nchar(self):
        """Return the number of characters in the matrix."""
        return self._matrix.shape[1]

    @property
    def ordered(self):
        """Return a list of the characters that are defined as ordered."""
        return self._ordered

    @ordered.setter
    def ordered(self, ordered):
        """Set a list of the characters that are defined as ordered."""
        self._ordered = ordered

    @property
    def taxa(self):
        """Return a list of the names of the taxa in the matrix."""
        return self._taxa

    @taxa.setter
    def taxa(self, taxa):
        """Set a list of the names of the taxa in the matrix."""
        self._taxa = taxa

    @property
    def description(self):
        """Return the description of the matrix."""
        return self._description

    @description.setter
    def description(self, description):
        """Set the description of the matrix."""
        self._description = description

    @property
    def missing(self):
        """Return the number of missing entries in the matrix."""
        return np.isnan(self._matrix).sum()

    @property
    def taxa_missing(self):
        """Return the number of missing entries per taxon in the matrix."""
        return np.sum(np.isnan(self._matrix), axis=1)

    @property
    def uninformative(self):
        """Return a list of the characters that contain uninformative data."""
        return self._uninformative

    @uninformative.setter
    def uninformative(self, uninformative):
        """Set a list of the characters that contain uninformative data."""
        self._uninformative = uninformative

    @property
    def missing_entries(self):
        """Return a list containing the frequency of missing entries for all
        characters."""
        missing_entries_all = np.sum(np.isnan(self.matrix), axis=0)
        missing = []
        for missing_entries in np.unique(missing_entries_all):
            frequency = np.sum(missing_entries_all == missing_entries)
            missing.append((frequency, missing_entries))
        return missing

    @property
    def uninf_entries(self):
        """Return a list containing the frequency of uninformative entries for
        the characters with invalid data."""
        uninf_all = np.sum(np.isfinite(self._matrix[:,list(self.uninformative)]),
                           axis=0)
        uninf = []
        for uninf_entries in np.unique(uninf_all):
            frequency = np.sum(uninf_all == uninf_entries)
            uninf.append((frequency, uninf_entries))
        return uninf

    def printable(self):
        """Return a version of the matrix array that contains printable
        characters."""
        return np.where(np.isnan(self._matrix), '?', np.nan_to_num(self._matrix).astype(int))

    def copy(self):
        """Return a copy of the current object."""
        taxeq = PyTaxEq()
        taxeq.matrix = self.matrix.copy()
        taxeq.ordered = self.ordered
        taxeq.description = self.description
        taxeq.taxa = self.taxa
        taxeq.uninformative = self.uninformative
        return taxeq

def equivalence(taxeq):
    """
    Calculate the equivalence states for a taxon-character matrix and the
    number of invalid comparisons.
    """
    equiv_array = np.zeros((taxeq.ntax, taxeq.ntax), dtype='|S1')
    invalid_pairs = 0

    for index in np.ndenumerate(equiv_array):
        taxon_b, taxon_a = index[0]

        if taxon_a == taxon_b:
            continue
        # Test whether equivalency of any form is present
        equiv_test_pre = np.subtract(taxeq.matrix[taxon_a], taxeq.matrix[taxon_b])
        equiv_test = np.unique(equiv_test_pre[~np.isnan(equiv_test_pre)])
        invalid_pairs += np.isnan(equiv_test_pre).sum()

        if len(equiv_test) == 0 or len(equiv_test) == 1 and equiv_test[0] == 0:
            result = np.subtract(np.isnan(taxeq.matrix[taxon_a]) + 0,
                                 np.isnan(taxeq.matrix[taxon_b]))
            val_range = max(result) - min(result)

            if val_range == 0:
                if np.all(~np.isnan(equiv_test_pre)):
                    # Taxa are actual equivalents (symmetric)
                    equiv_array[taxon_a, taxon_b] = 'A'
                else:
                    # Taxa are potential equivalents (symmetric)
                    equiv_array[taxon_a, taxon_b] = 'B'
            elif val_range == 1:
                # Taxa are potential equivalents (asymmetric one way)
                if max(result) == 0:
                    equiv_array[taxon_a, taxon_b] = 'C'
                else:
                    equiv_array[taxon_a, taxon_b] = 'E'
            else:
                # Taxa are potential equivalents (asymmetric both ways)
                equiv_array[taxon_a, taxon_b] = 'D'

    return {'array': equiv_array, 'invalid_pairs': invalid_pairs}

def suffixes(taxeq, equiv_array):
    """
    Analyze an equivalence array to determine if suffixes should be applied.
    Any comparison to a taxon with no informative data (e.g. "!") or that must
    originate from the same node (e.g. "*", excluding arbitrary resolutions)
    will be marked as such.
    """
    addition_arr = np.zeros((taxeq.ntax, taxeq.ntax), dtype='|S1')
    # These taxa are subsumed in their respective index taxa...
    tests = np.argwhere(np.any([equiv_array == 'A', equiv_array == 'B',
                                equiv_array == 'C'], axis=0))
    indices = np.arange(taxeq.ntax)
    for idx, equiv in tests:
        # Check if equivalent contains any informative data
        if np.all(np.isnan(taxeq.matrix[equiv])):
            addition_arr[idx, equiv] += '!'
        else:
            screen = ~np.any([indices == idx, indices == equiv], axis=0)
            if ~np.any(equiv_array[:,idx][screen] == 'C'):
                addition_arr[idx, equiv] += '*'

    return addition_arr

def compute(taxeq):
    """
    Perform a test of safe taxonomic equivalence on a given dataset. Returns a
    array of equivalence indices and suffixes, a count of the total number of
    invalid comparisons, and a set of indices representing taxa that can be
    safely removed.
    """
    equivs = equivalence(taxeq)
    extras = suffixes(taxeq, equivs['array'])
    deletes = set()
    for t_id, tax_eq in enumerate(equivs['array']):
        if t_id in deletes:
            continue
        deletes.update(np.where(np.any(
            [tax_eq == 'A', tax_eq == 'B', tax_eq == 'C'], axis=0))[0].tolist())
    return {'equivs': np.core.defchararray.add(equivs['array'], extras),
            'invalid_pairs': equivs['invalid_pairs'], 'deletes': deletes}

def check_info(taxeq):
    """
    Check for and reports uninformative data in the matrix. This function will
    also perform in-place modification of the matrix, replacing any
    uninformative data with NaNs.
    """
    check_mtx = taxeq.matrix.T
    dispersion, recode, replace = [], [], []

    for char_n, char_states in enumerate(check_mtx):
        counts = Counter(char_states[~np.isnan(char_states)])
        # Check for single character uninformative states
        if len(counts) < 2:
            replace.append({'character': char_n})
            taxeq.uninformative.add(char_n)
            check_mtx[char_n] = np.nan
        # Check for binary uninformative characters
        elif len(counts) == 2:
            for char_state, count in counts.items():
                if count == 1:
                    replace.append({'character': char_n})
                    dispersion.append(np.isfinite(char_states).sum())
                    taxeq.uninformative.add(char_n)
                    check_mtx[char_n] = np.nan
        else:
            # Check for multistate ordered uninformative characters
            if char_n + 1 in taxeq.ordered:
                if counts[len(counts) - 1] == 1:
                    tax_n = np.where(char_states == len(counts) - 1)[0][0]
                    recode.append({'character': char_n, 'taxon': tax_n,
                                   'from': len(counts) - 1,
                                   'to': len(counts) - 2})
                    check_mtx[char_n, tax_n] -= 1
            # Check for multistate unordered uninformative characters
            else:
                if not counts.values().count(1):
                    continue
                elif counts.values().count(1) > 1:
                    replace.append({'character': char_n})
                    taxeq.uninformative.add(char_n)
                    check_mtx[char_n] = np.nan
                else:
                    char_state = counts.most_common()[-1][0]
                    tax_n = np.where(char_states == char_state)[0][0]
                    recode.append({'character': char_n, 'taxon': tax_n,
                                'from': int(char_state)})
                    check_mtx[char_n, tax_n] = np.nan

    uninformative = {'dispersion': Counter(dispersion), 'matrix': check_mtx,
                     'recode': recode, 'replace': replace}
    return uninformative

def report(taxeq, out_format, out_file):
    """
    Produce a report detailing the scope for safe taxonomic reduction.
    """
    equiv_m = compute(taxeq)
    matrices = [taxeq.copy()]
    uninformative = check_info(taxeq)
    matrices.append(taxeq.copy())

    if uninformative['recode'] or uninformative['replace']:
        equiv_u = compute(taxeq)
    else:
        equiv_u = []

    if out_format == 'txt':
        print "This doesn't work yet!"
        return
    elif out_format == 'html':
        template = env.get_template('pytaxeq.jinja.html')

    with open(out_file, 'wb') as report_file:
        report_file.write(
            template.render(matrices=matrices, equiv_missing=equiv_m,
                            equiv_uninf=equiv_u, uninformative=uninformative))

if __name__ == "__main__":
    import argparse
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("infile", help="name of the file to analyze")
    parser.add_argument("-o", "--outfile", help="name of the output file")
    parser.add_argument("-f", "--format", choices=['txt', 'html'],
                        default='html', help="output format (default: html)")
    args = parser.parse_args()
    outfile = args.outfile or "{0}.{1}".format(os.path.splitext(os.path.split(args.infile)[-1])[0], args.format)

    try:
        taxeq_matrix = PyTaxEq(args.infile)
        report(taxeq_matrix, args.format, outfile)
    except IOError as err:
        print "Error: {0} '{1}'".format(err.strerror, err.filename)
