const DojoApp = artifacts.require("DojoApp");

module.exports = function(deployer) {
  deployer.deploy(DojoApp);
};