# -*- coding: utf-8 -*-
from projects.FixMatch import exec as FixMatch
from projects.FixMatch_UBPL import exec as FixMatch_UBPL


def execute():
    dataParams = [["CIFAR10", 40], ["CIFAR10", 250], ["CIFAR10", 4000],
                  ["CIFAR100", 400], ["CIFAR100", 2500], ["CIFAR100", 10000],
                  ["SVHN", 40], ["SVHN", 250], ["SVHN", 1000]]
    for dataParam in dataParams:
        dataset, trainCount_labeled = dataParam

        # FixMatch
        FixMatch("FixMatch", {"dataset": dataset, "trainCount_labeled": trainCount_labeled, "useConsistency": False})

        # FixMatch+UBPL
        FixMatch_UBPL("FixMatch_UBPL", {"dataset": dataset, "trainCount_labeled": trainCount_labeled, "useFDL": True})
    pass


if __name__ == "__main__":
    execute()
