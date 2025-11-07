"""RunSet tests"""

import os

from cmdstanpy import _TMPDIR
from cmdstanpy.cmdstan_args import CmdStanArgs, SamplerArgs
from cmdstanpy.stanfit import RunSet
from cmdstanpy.utils import EXTENSION

HERE = os.path.dirname(os.path.abspath(__file__))
DATAFILES_PATH = os.path.join(HERE, 'data')


def test_check_repr() -> None:
    exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    sampler_args = SamplerArgs()
    chain_ids = [1, 2, 3, 4]  # default
    cmdstan_args = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=chain_ids,
        data=jdata,
        method_args=sampler_args,
    )
    runset = RunSet(args=cmdstan_args, chains=4)
    assert 'RunSet: chains=4' in repr(runset)
    assert 'method=sample' in repr(runset)
    assert 'retcodes=[-1, -1, -1, -1]' in repr(runset)
    assert 'csv_file' in repr(runset)
    assert 'console_msgs' in repr(runset)
    assert 'diagnostics_file' not in repr(runset)


def test_check_retcodes() -> None:
    exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    sampler_args = SamplerArgs()
    chain_ids = [1, 2, 3, 4]  # default
    cmdstan_args = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=chain_ids,
        data=jdata,
        method_args=sampler_args,
    )
    runset = RunSet(args=cmdstan_args, chains=4)

    retcodes = runset._retcodes
    assert 4 == len(retcodes)
    for i in range(len(retcodes)):
        assert -1 == runset._retcode(i)
    runset._set_retcode(0, 0)
    assert 0 == runset._retcode(0)
    for i in range(1, len(retcodes)):
        assert -1 == runset._retcode(i)
    assert not runset._check_retcodes()
    for i in range(1, len(retcodes)):
        runset._set_retcode(i, 0)
    assert runset._check_retcodes()


def test_get_err_msgs() -> None:
    exe = os.path.join(DATAFILES_PATH, 'logistic' + EXTENSION)
    rdata = os.path.join(DATAFILES_PATH, 'logistic.missing_data.R')
    sampler_args = SamplerArgs()
    chain_ids = [1, 2, 3]
    cmdstan_args = CmdStanArgs(
        model_name='logistic',
        model_exe=exe,
        chain_ids=chain_ids,
        data=rdata,
        method_args=sampler_args,
    )
    runset = RunSet(args=cmdstan_args, chains=3, chain_ids=chain_ids)
    for i in range(3):
        runset._set_retcode(i, 70)
        stdout_file = 'chain-' + str(i + 1) + '-missing-data-stdout.txt'
        path = os.path.join(DATAFILES_PATH, stdout_file)
        runset._stdout_files[i] = path
    errs = runset.get_err_msgs()
    assert 'Exception: variable does not exist' in errs


def test_output_filenames_one_proc_per_chain() -> None:
    exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    sampler_args = SamplerArgs()
    chain_ids = [1, 2, 3, 4]
    cmdstan_args = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=chain_ids,
        data=jdata,
        method_args=sampler_args,
    )
    runset = RunSet(args=cmdstan_args, chains=4, one_process_per_chain=True)

    assert all("bernoulli-" in csv_file for csv_file in runset.csv_files)
    assert all(
        csv_file.endswith(f"_{id}.csv")
        for id, csv_file in zip(chain_ids, runset.csv_files)
    )
    assert len(runset.stdout_files) == len(chain_ids)
    assert all(
        stdout_file.endswith(f"_stdout_{id}.txt")
        for id, stdout_file in zip(chain_ids, runset.stdout_files)
    )

    cmdstan_args_other_files = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=chain_ids,
        data=jdata,
        method_args=sampler_args,
        save_latent_dynamics=True,
        save_profile=True,
    )
    runset_other_files = RunSet(
        args=cmdstan_args_other_files, chains=4, one_process_per_chain=True
    )
    assert len(runset_other_files.diagnostic_files) == len(chain_ids)
    assert all(
        diag_file.endswith(f"_diagnostic_{id}.csv")
        for id, diag_file in zip(chain_ids, runset_other_files.diagnostic_files)
    )

    assert len(runset_other_files.profile_files) == len(chain_ids)
    assert all(
        prof_file.endswith(f"_profile_{id}.csv")
        for id, prof_file in zip(chain_ids, runset_other_files.profile_files)
    )


def test_output_filenames_threading() -> None:
    exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    sampler_args = SamplerArgs()
    chain_ids = [1, 2, 3, 4]
    cmdstan_args = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=chain_ids,
        data=jdata,
        method_args=sampler_args,
    )
    runset = RunSet(args=cmdstan_args, chains=4, one_process_per_chain=False)

    assert all("bernoulli-" in csv_file for csv_file in runset.csv_files)
    assert all(
        csv_file.endswith(f"_{id}.csv")
        for id, csv_file in zip(chain_ids, runset.csv_files)
    )
    assert len(runset.stdout_files) == 1
    assert runset.stdout_files[0].endswith("_stdout.txt")

    cmdstan_args_other_files = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=chain_ids,
        data=jdata,
        method_args=sampler_args,
        save_latent_dynamics=True,
        save_profile=True,
    )
    runset_other_files = RunSet(
        args=cmdstan_args_other_files, chains=4, one_process_per_chain=False
    )
    assert len(runset_other_files.diagnostic_files) == len(chain_ids)
    assert all(
        diag_file.endswith(f"_diagnostic_{id}.csv")
        for id, diag_file in zip(chain_ids, runset_other_files.diagnostic_files)
    )

    assert len(runset_other_files.profile_files) == 1
    assert runset_other_files.profile_files[0].endswith("_profile.csv")


def test_commands() -> None:
    exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    sampler_args = SamplerArgs()
    chain_ids = [1, 2, 3, 4]
    cmdstan_args = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=chain_ids,
        data=jdata,
        method_args=sampler_args,
    )
    runset = RunSet(args=cmdstan_args, chains=4)
    assert 'id=1' in runset.cmd(0)
    assert 'id=4' in runset.cmd(3)


def test_save_latent_dynamics() -> None:
    exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    sampler_args = SamplerArgs()
    chain_ids = [1, 2, 3, 4]
    cmdstan_args = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=chain_ids,
        data=jdata,
        method_args=sampler_args,
        save_latent_dynamics=True,
    )
    runset = RunSet(args=cmdstan_args, chains=4)
    assert _TMPDIR in runset.diagnostic_files[0]

    cmdstan_args = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=chain_ids,
        data=jdata,
        method_args=sampler_args,
        save_latent_dynamics=True,
        output_dir=os.path.abspath('.'),
    )
    runset = RunSet(args=cmdstan_args, chains=4)
    assert os.path.abspath('.') in runset.diagnostic_files[0]


def test_chain_ids() -> None:
    exe = os.path.join(DATAFILES_PATH, 'bernoulli' + EXTENSION)
    jdata = os.path.join(DATAFILES_PATH, 'bernoulli.data.json')
    sampler_args = SamplerArgs()
    chain_ids = [11, 12, 13, 14]
    cmdstan_args = CmdStanArgs(
        model_name='bernoulli',
        model_exe=exe,
        chain_ids=chain_ids,
        data=jdata,
        method_args=sampler_args,
    )
    runset = RunSet(args=cmdstan_args, chains=4, chain_ids=chain_ids)
    assert 'id=11' in runset.cmd(0)
    assert '_11.csv' in runset._csv_files[0]
    assert 'id=14' in runset.cmd(3)
    assert '_14.csv' in runset._csv_files[3]
