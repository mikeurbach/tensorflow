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
    name = "rtl_ops_inc_gen",
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "ir/rtl_ops.td",
    tbl_outs = [
        ("-gen-op-decls", "ir/rtl_ops.h.inc"),
        ("-gen-op-defs", "ir/rtl_ops.cc.inc"),
    ],
    td_srcs = [
        "@llvm-project//mlir:OpBaseTdFiles",
        "@llvm-project//mlir:include/mlir/Interfaces/SideEffects.td",
    ],
)

gentbl(
    name = "rtl_process_dataflow_inc_gen",
    tblgen = "@llvm-project//mlir:mlir-tblgen",
    td_file = "transforms/rtl_process_dataflow.td",
    tbl_outs = [
        ("-gen-rewriters", "transforms/rtl_process_dataflow.cc.inc"),
    ],
    td_srcs = [
        "ir/rtl_ops.td",
        "@llvm-project//mlir:OpBaseTdFiles",
    ],
)

cc_library(
    name = "rtl",
    srcs = [
        "ir/rtl_ops.h.inc",
        "ir/rtl_ops.cc.inc",
        "ir/rtl_ops.cc",
    ],
    hdrs = [
        "ir/rtl_ops.h",
    ],
    includes = ["include"],
    deps = [
        ":rtl_ops_inc_gen",
        "@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)

cc_library(
    name = "rtl_dialect_registration",
    srcs = [
        "ir/dialect_registration.cc",
    ],
    deps = [
        ":rtl",
	"@llvm-project//mlir:IR",
    ],
    alwayslink = 1,
)

cc_library(
    name = "rtl_process_dataflow",
    srcs = [
        "transforms/rtl_process_dataflow.cc.inc",
        "transforms/rtl_process_dataflow.cc",
    ],
    deps = [
        ":rtl",
        ":rtl_dialect_registration",
	":rtl_process_dataflow_inc_gen",
	"//tensorflow/compiler/mlir/dataflow",
	"//tensorflow/compiler/mlir/dataflow:dataflow_dialect_registration",
	"//tensorflow/compiler/mlir/xla:hlo",
	"//tensorflow/compiler/mlir/xla:xla_dialect_registration",
        "@llvm-project//llvm:support",
        "@llvm-project//mlir:Dialect",
        "@llvm-project//mlir:IR",
        "@llvm-project//mlir:Pass",
        "@llvm-project//mlir:Transforms",
    ],
    alwayslink = 1,
)

tf_cc_binary(
    name = "rtl-opt",
    deps = [
        ":rtl_process_dataflow",
        "//tensorflow/compiler/mlir:tf_mlir_opt_main",
    ],
)

compilation_database(
    name = "compilation_database",
    targets = [
      "rtl_process_dataflow",
    ],
)
