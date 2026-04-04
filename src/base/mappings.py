from src.base.schemas import Domains, Images, Problems, Prompts

DOMAIN_PROMPTS = {
    Domains.RING_AND_PEG: Prompts.RING_AND_PEG_DOMAIN,
    Domains.NEEDLE_TRANSFER: Prompts.NEEDLE_TRANSFER_DOMAIN,
}

PROBLEM_PROMPTS = {
    Problems.RING_AND_PEG_1: Prompts.RING_AND_PEG_1,
    Problems.RING_AND_PEG_2: Prompts.RING_AND_PEG_2,
    Problems.RING_AND_PEG_3: Prompts.RING_AND_PEG_3,
    Problems.RING_AND_PEG_4: Prompts.RING_AND_PEG_4,
    Problems.RING_AND_PEG_5: Prompts.RING_AND_PEG_5,
    Problems.NEEDLE_TRANSFER_1: Prompts.NEEDLE_TRANSFER_1,
    Problems.NEEDLE_TRANSFER_2: Prompts.NEEDLE_TRANSFER_2,
}

ACTION_SCHEMA_PROMPTS = {
    Domains.RING_AND_PEG: Prompts.ACTION_SCHEMA_RING_AND_PEG,
    Domains.NEEDLE_TRANSFER: Prompts.ACTION_SCHEMA_NEEDLE_TRANSFER,
}

IMAGES = {
    Problems.RING_AND_PEG_1: Images.RING_AND_PEG_1,
    Problems.RING_AND_PEG_2: Images.RING_AND_PEG_2,
    Problems.RING_AND_PEG_3: Images.RING_AND_PEG_3,
    Problems.RING_AND_PEG_4: Images.RING_AND_PEG_4,
    Problems.RING_AND_PEG_5: Images.RING_AND_PEG_5,
    Problems.NEEDLE_TRANSFER_1: Images.NEEDLE_TRANSFER_1,
    Problems.NEEDLE_TRANSFER_2: Images.NEEDLE_TRANSFER_2,
}
