load("//tensorflow:tensorflow.bzl", "tf_cc_binary")
load("//third_party/mlir:tblgen.bzl", "gentbl")
load("@com_grail_bazel_compdb//:aspects.bzl", "compilation_database")

package(
    default_visibility = [":friends"],
    licenses = ["notice"],  # Apache 2.0
)

package_group(
    name = "friends",
    packages = [
        "//tensorflow/compiler/mlir/...",
    ],
)

gentbl(
    name = "dataflow_ops_inc_gen",
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/dataflow_ops.td",
    tbl_outs = [
        ("-gen-op-decls", "ir/dataflow_ops.h.inc"),
        ("-gen-op-defs", "ir/dataflow_ops.cc.inc"),
    ],
    td_srcs = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffects.td",
    ],
)

cc_library(
    name = "dataflow",
    srcs = [
        "ir/dataflow_ops.h.inc",
        "ir/dataflow_ops.cc.inc",
        "ir/dataflow_ops.cc",
    ],
    hdrs = [
        "ir/dataflow_ops.h",
    ],
    includes = ["include"],
    deps = [
        ":dataflow_ops_inc_gen",
	"//tensorflow/compiler/mlir/xla:hlo",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)

cc_library(
    name = "dataflow_dialect_registration",
    srcs = [
        "ir/dialect_registration.cc",
    ],
    deps = [
        ":dataflow",
	"@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)

cc_library(
    name = "dataflow_legalize_hlo",
    srcs = [
        "transforms/dataflow_legalize_hlo.cc",
    ],
    deps = [
        ":dataflow",
        ":dataflow_dialect_registration",
	"//tensorflow/compiler/mlir/xla:hlo",
	"//tensorflow/compiler/mlir/xla:xla_dialect_registration",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

tf_cc_binary(
    name = "dataflow-opt",
    deps = [
        ":dataflow_legalize_hlo",
        "//tensorflow/compiler/mlir:tf_mlir_opt_main",
    ],
)

compilation_database(
    name = "compilation_database",
    targets = [
      "dataflow_legalize_hlo",
    ],
)
