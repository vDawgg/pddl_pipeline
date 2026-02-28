## Domain description

An agent is tasked with picking, equipping, crafting and recalling (putting an item back into the inventory) a set of items. Crafting an item in a crafting-slot leads to the original input item being consumed and turns the slot into the newly crafted item. The only crafting recipe currently available to the agent is crafting a plank from a log. All items including the crafting-slots can be equiped and carried in the inventory.

The available actions are:
- recall(item, agent)
- move(from, to)
- craftplank(craft-table, agent, item)
- equip(item, agent)
- pick(item, location)
