type-check:
	@mypy .

test:
    # We add the `-I` flag to only use the installed package and not use local modules.
    # Note that we cannot add it to all tests because of the implemented import logic.
    # See PR #49 for more details.
    # TODO: Find better solution.
	@python3 -I -m unittest discover -s tests/ -p "*_test.py"
	@python3 -I -m unittest discover -s distributed_shampoo/tests/ -p "*_test.py"
	@python3 -m unittest discover -s distributed_shampoo/utils/tests/ -p "*_test.py"
	@python3 -m unittest discover -s distributed_shampoo/gpu_tests/ -p "*_test.py"
	@python3 -m unittest distributed_shampoo/utils/gpu_tests/shampoo_dist_utils_test.py
	@torchrun --standalone --nnodes=1 --nproc_per_node=2 -m unittest distributed_shampoo/utils/gpu_tests/shampoo_ddp_distributor_test.py

test-gpu:
	@python3 -m unittest discover -s distributed_shampoo/gpu_tests/ -p "*_test.py"
	@python3 -m unittest distributed_shampoo/utils/gpu_tests/shampoo_dist_utils_test.py

test-multi-gpu:
	@torchrun --standalone --nnodes=1 --nproc_per_node=2 -m unittest discover -s distributed_shampoo/gpu_tests/ -p "*_test.py"
	@torchrun --standalone --nnodes=1 --nproc_per_node=2 -m unittest distributed_shampoo/utils/gpu_tests/shampoo_dist_utils_test.py
	@torchrun --standalone --nnodes=1 --nproc_per_node=2 -m unittest distributed_shampoo/utils/gpu_tests/shampoo_ddp_distributor_test.py
	@torchrun --standalone --nnodes=1 --nproc_per_node=2 -m unittest distributed_shampoo/utils/gpu_tests/shampoo_fsdp_distributor_test.py
	@torchrun --standalone --nnodes=1 --nproc_per_node=2 -m unittest distributed_shampoo/utils/gpu_tests/shampoo_fully_shard_distributor_test.py
	@torchrun --standalone --nnodes=1 --nproc_per_node=4 -m unittest distributed_shampoo/utils/gpu_tests/shampoo_hsdp_distributor_test.py
	@torchrun --standalone --nnodes=1 --nproc_per_node=4 -m unittest distributed_shampoo/utils/gpu_tests/shampoo_fsdp_utils_test.py
