paths = dict(
    root_dir="/export/home2/kningtg/WORKSPACE/moment-retrieval/query-moment",
    data_dir="/export/home2/kningtg/WORKSPACE/moment-retrieval/data-bin/raw",
    cache_dir="/export/home2/kningtg/WORKSPACE/moment-retrieval/data-bin/cache",
    work_dir="${paths.root_dir}/work_dir/${data.dataset}/${flags.exp}",
)

flags = dict(debug=False, ddp=False, amp=False, train=True, wandb=False, seed=3147)
