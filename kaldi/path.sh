export KALDI_ROOT=/opt/kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "missing $KALDI_ROOT/tools/config/common_path.sh" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
