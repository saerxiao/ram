local utils = {}

function utils.get_kwarg(kwargs, name, default)
  if kwargs == nil then kwargs = {} end
  if kwargs[name] == nil and default == nil then
    assert(false, string.format('"%s" expected and not given', name))
  elseif kwargs[name] == nil then
    return default
  else
    return kwargs[name]
  end
end

return utils