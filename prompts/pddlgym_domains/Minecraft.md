## Domain description

An agent is tasked with picking, equipping, crafting and recalling (putting an item back into the inventory) a set of items. Crafting an item in a crafting-slot leads to the original input item being consumed and turns the slot into the newly crafted item. The only crafting recipe currently available to the agent is crafting a plank from a log. All items including the crafting-slots can be equiped and carried in the inventory.

The available actions are:
- recall
    - Makes the agent put the item back into inventory
- move
    - Move the agent to a new position
- craftplank
    - Lets the agent craft a plank from a crafting-table and an item
- equip
    - Makes the agent equip an item
- pick
    - Makes the agent pick up an item at a specific location
