#!/usr/bin/env python
import os
import sys

galaxy_path = os.environ.get('GALAXY', None)
if galaxy_path is None:
    galaxy_path = os.getcwd()

new_path = os.path.join(os.path.abspath(galaxy_path), 'lib')
sys.path.insert(0, new_path)

from six.moves import configparser
from galaxy.model.orm.engine_factory import build_engine
from galaxy.model.mapping import MetaData
from sqlalchemy.sql import and_
from sqlalchemy.orm import (scoped_session, sessionmaker)
import galaxy.model

import argparse
import ast
import decimal
import pkg_resources
import json
import datetime
import time
import re
from curses import wrapper
import pandas as pd
import logging

log = logging.getLogger(__name__)


class ProgressBar(object):
    STYLE = '%(bar)s %(current)d/%(total)d ' + \
            '(%(percent)3d%%) %(remaining)d remaining'

    def __init__(self, total, width=40, symbol='=',
                 output=sys.stderr):
        assert len(symbol) == 1

        self.total = total
        self.width = width
        self.symbol = symbol
        self.output = output
        self.fmt = re.sub(r'(?P<name>%\(.+?\))d',
                          r'\g<name>%dd' % len(str(total)),
                          self.STYLE)

        self.current = 0

    def __call__(self):
        percent = self.current / float(self.total)
        size = int(self.width * percent)
        remaining = self.total - self.current
        bar = '[' + self.symbol * size + ' ' * (self.width - size) + ']'

        args = {
            'total': self.total,
            'bar': bar,
            'current': self.current,
            'percent': percent * 100,
            'remaining': remaining
        }
        return self.fmt % args

    def done(self):
        self.current = self.total
        self()
        return ''


class JobInfo(object):

    lines = dict()
    job_data = []
    process_times = []

    def __init__(self, stdscr, args):
        self.stdscr = stdscr
        self.height, self.width = stdscr.getmaxyx()
        self.args = args
        self.config = self._parse_config(args.config)
        self.dburi = self.conf_parser.get('app:main',
                                          'database_connection')
        if self.dburi is None:
            self.dburi = self.conf_parser.get('galaxy',
                                              'database_connection')
        if self.dburi is None:
            self.dburi = 'sqlite://./database/universe.sqlite'
        text = 'Getting job data from {} for {}'.format(self.dburi, args.toolid)
        self._addline(0, 0, text)
        self.outfile = args.outfile
        self.toolid = args.toolid
        self.fh = open(self.outfile, 'w')
        self._db()
        # self.process()

    def _parse_config(self, configfile):
        self.conf_parser = configparser.ConfigParser()
        self.conf_parser.read(configfile)

    def _clear(self):
        self.stdscr.clear()

    def _addline(self, row, col, text, refresh=True):
        row = str(row)
        col = str(col)
        if row in self.lines:
            self.lines[row][col] = text
        else:
            self.lines[row] = dict()
            self.lines[row][col] = text
        self._curse(refresh)

    def _db(self):
        '''Initialize the database file.'''
        dialect_to_egg = dict(sqlite='pysqlite>=2',
                              postgres='psycopg2',
                              mysql='MySQL_python')
        dialect = (self.dburi.split(':', 1))[0]
        try:
            egg = dialect_to_egg[dialect]
            try:
                pkg_resources.require(egg)
                log.debug('Lodaded egg %s for %s dialect' % (egg, dialect))
            except Exception:
                # If the module is in the path elsewhere
                # (i.e. non-egg), it'll still load.
                log.debug('Egg not found, attempting %s anyway' % (dialect))
        except KeyError:
            # Let this go, it could possibly work with db's we don't support.
            log.debug('Unknown SQLAlchemy database dialect: %s' % dialect)
        # Initialize the database connection.
        engine = build_engine(self.dburi, dict())
        MetaData(bind=engine)
        Session = scoped_session(sessionmaker(bind=engine,
                                              autoflush=False,
                                              autocommit=True))
        self.db = Session

    def _curse(self, refresh=True):
        # self.stdscr.clear()
        for row in self.lines:
            for col in self.lines[row]:
                row = int(row)
                col = int(col)
                self.stdscr.addstr(row, col, self.lines[str(row)][str(col)])
                self.stdscr.clrtoeol()
        if refresh:
            self.stdscr.refresh()

    def _normalize_data(self):
        self._addline(5, 0, 'Normalizing job data')
        self.normalized_data = pd.io.json.json_normalize(self.job_data)
        self._addline(5, 0, 'Normalizing job data ... done')
        self._addline(6, 0, 'Filtering out null rows')
        self.normalized_data.set_index('id')
        df = self.normalized_data['runtime_seconds'].notnull()
        self.normalized_data = self.normalized_data[df]

    def _flatten_dict(self, job, dictionary, delimiter):
        if isinstance(dictionary, dict):
            for key in dictionary:
                if isinstance(dictionary[key], dict):
                    job = self._flatten_dict(job,
                                             dictionary[key],
                                             delimiter + '.' + key)
                else:
                    job[delimiter + '.' + key] = dictionary[key]
        else:
            pass
        return job

    def get_dataset_info(self, job):
        retval = dict()
        input_datasets = job.input_datasets
        if input_datasets is None:
            return retval
        for input_dataset in input_datasets:
            if input_dataset is None:
                continue
            hda = input_dataset.dataset
            if hda is None:
                continue
            dataset = hda.dataset
            if dataset is None:
                continue
            retval[input_dataset.name] = int(dataset.file_size)
            retval['%s_filetype' % input_dataset.name] = hda.extension
        return retval

    def parse_metrics(self, metrics):
        retval = dict()
        if metrics is None:
            return retval
        for metric in metrics:
            if metric is None:
                continue
            if isinstance(metric.metric_value, decimal.Decimal):
                metric.metric_value = float(metric.metric_value)
            retval.update({metric.metric_name: metric.metric_value})
        return retval

    def parse_parameters(self, parameters):
        retval = dict()
        if parameters is None:
            return retval
        for parameter in parameters:
            if parameter is None:
                continue
            try:
                evaled_value = ast.literal_eval(parameter.value)
                param_key = 'parameters.%s' % parameter.name
                if isinstance(evaled_value, dict):
                    retval = self._flatten_dict(retval,
                                                evaled_value,
                                                param_key)
                else:
                    retval[param_key] = parameter.value
            except Exception:
                raise
        return retval

    def save(self):
        outfile = self.args.outfile
        with open(outfile, 'w') as fh:
            if outfile.endswith('.json'):
                fh.write(json.dumps(self.job_data, indent=4, sort_keys=True))
            else:
                self._normalize_data()
                fh.write(self.normalized_data.to_csv())

    def extract_data_from_db(self):
        query = self.db.query(galaxy.model.Job)
        if '%' in self.toolid or '_' in self.toolid:
            pattern = and_(galaxy.model.Job.table.c.tool_id.like(self.toolid),
                           galaxy.model.Job.table.c.state == 'ok')
            query = query.filter(pattern)
        else:
            pattern = and_(galaxy.model.Job.table.c.tool_id == self.toolid,
                           galaxy.model.Job.table.c.state == 'ok')
            query = query.filter(pattern)
        results = query.all()
        self.record_count = len(results)
        self.progress = ProgressBar(self.record_count)
        self.current_record = 0
        self.elapsed_time = 0
        for result in results:
            if len(self.process_times) > 25:
                self.process_times.pop()
            seconds_remaining = 999999.999
            time_avg = 0.0
            if self.current_record != 0:
                time_avg = sum(self.process_times) / \
                    float(len(self.process_times))
                seconds_remaining = time_avg * \
                    (self.record_count - self.current_record)
            self._addline(2, 0, self.progress(), False)
            avg = 'Average time per record: {:.4f} seconds'.format(time_avg)
            self._addline(4, 0, avg, False)
            remains = datetime.timedelta(seconds=seconds_remaining)
            self._addline(5, 0, 'Estimated time remaining: {}'.format(remains))
            start_time = time.time()
            if result is None:
                continue
            current_job = {'command_line': result.command_line,
                           'copied_from_job_id': result.copied_from_job_id,
                           'create_time': str(result.create_time),
                           'dependencies': result.dependencies,
                           'destination_id': result.destination_id,
                           'destination_params': result.destination_params,
                           'exit_code': result.exit_code,
                           'handler': result.handler,
                           'history_id': result.history_id,
                           'id': result.id,
                           'imported': result.imported,
                           'info': result.info,
                           'job_runner_name': result.job_runner_name,
                           'library_folder_id': result.library_folder_id,
                           'object_store_id': result.object_store_id,
                           'param_filename': result.param_filename,
                           'runner_name': result.runner_name,
                           'session_id': result.session_id,
                           'state': result.state,
                           'tool_id': result.tool_id,
                           'tool_version': result.tool_version,
                           'traceback': result.traceback,
                           'update_time': str(result.update_time),
                           'user_id': result.user_id}
            current_job.update(self.parse_metrics(result.metrics))
            current_job.update(self.parse_parameters(result.get_parameters()))
            current_job.update(self.get_dataset_info(result))
            self.job_data.append(current_job)
            finish_time = time.time()
            time_elapsed = finish_time - start_time
            self.process_times.append(time_elapsed)
            self.elapsed_time += (time_elapsed)
            self.current_record += 1
            self.progress.current = self.current_record
        result = 'Processed {} records in {}'
        retval = result.format(self.record_count,
                               datetime.timedelta(seconds=self.elapsed_time))
        self._addline(5, 0, retval)
        self.return_code = 0


def main(stdscr, args):
    jobinfo = JobInfo(stdscr, args)
    jobinfo._clear()
    return_code = jobinfo.extract_data_from_db()
    jobinfo.save()
    return jobinfo


if __name__ == '__main__':
    description = 'Extract job data for a specific tool ID from a galaxy '
    description += 'database. For best results, run from your galaxy '
    description += "instance's root path."
    epilog = 'Tool ID wildcards supported are % for multiple characters '
    epilog += 'and _ for single characters.'
    parser = argparse.ArgumentParser(description=description,
                                     epilog=epilog)
    parser.add_argument('-c',
                        '--config',
                        dest='config',
                        action='store',
                        default='config/galaxy.yml')
    parser.add_argument('-t',
                        '--toolid',
                        dest='toolid',
                        action='store',
                        required=True)
    parser.add_argument('-o',
                        '--outfile',
                        dest='outfile',
                        action='store',
                        default='job_data.json',
                        help='Output file, extension determines format.')
    args = parser.parse_args()
    result = wrapper(main, args)
    previous_newline = False
    for n in range(0, result.height):
        line = result.stdscr.instr(n, 0)
        if len(line.strip()) == 0:
            if not previous_newline:
                print('\n')
            previous_newline = True
        else:
            print(result.stdscr.instr(n, 0).decode('utf-8'))
            previous_newline = False
    exit(result.return_code)
